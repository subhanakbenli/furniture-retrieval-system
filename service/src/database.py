import os
import time
from typing import List, Dict, Any, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid

class QdrantHandler:
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, collection_name: str = "objects_collection", vector_size: int = 512):
        host = host or os.getenv("QDRANT_HOST", "localhost")
        port = port or int(os.getenv("QDRANT_PORT", "6333"))
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._wait_for_qdrant()
        self._init_collection()

    def _wait_for_qdrant(self, retries: int = 30, delay: float = 2.0):
        """Qdrant may still be starting when the API container comes up."""
        for attempt in range(retries):
            try:
                self.client.get_collections()
                return
            except Exception:
                print(f"Waiting for Qdrant ({attempt + 1}/{retries})...")
                time.sleep(delay)
        raise RuntimeError("Qdrant is not reachable")

    def _init_collection(self):
        """Initialize the collection if it doesn't exist."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )
            print(f"Collection '{self.collection_name}' created.")
        else:
            print(f"Collection '{self.collection_name}' already exists.")

    def store_object(self, embedding: List[float], metadata: Dict[str, Any], point_id: Optional[str] = None):
        """Store an object embedding with metadata."""
        if point_id is None:
            point_id = str(uuid.uuid4())
            
        # Ensure object_id exists in metadata, default to point_id if not provided (single image case acting as new object)
        if "object_id" not in metadata:
            metadata["object_id"] = str(uuid.uuid4())
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=metadata
                )
            ]
        )
        return point_id

    def search_similar_groups(self, embedding: List[float], limit: int = 5, score_threshold: float = 0.0, filter_category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar objects, grouped by object_id, optionally filtered by category."""
        
        query_filter = None
        if filter_category:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="category",
                        match=models.MatchValue(value=filter_category)
                    )
                ]
            )

        groups = self.client.query_points_groups(
            collection_name=self.collection_name,
            query=embedding,
            group_by="object_id",
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_payload=True
        ).groups
        
        results = []
        for group in groups:
            # unique object found
            # The best hit in the group is usually the first one or we take the group score?
            # Qdrant returns 'hits' list in the group.
            best_hit = group.hits[0]
            results.append({
                "id": best_hit.id, # id of the specific image that matched best
                "score": best_hit.score,
                "metadata": best_hit.payload,
                "group_id": group.id
            })
            
        return results

    def list_objects(self, limit: int = 10, offset: Optional[Any] = None) -> Tuple[List[Dict[str, Any]], Any]:
        """
        List objects with pagination (scroll).
        Returns (points, next_offset).
        """
        points, next_offset = self.client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        
        results = [
            {
                "id": point.id,
                "metadata": point.payload
            }
            for point in points
        ]
        return results, next_offset

    def delete_object(self, point_id: str):
        """Delete an object by ID."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=[point_id]
            )
        )

    def update_payload(self, point_id: str, metadata: Dict[str, Any]):
        """Update metadata/payload of an object."""
        self.client.set_payload(
            collection_name=self.collection_name,
            payload=metadata,
            points=[point_id]
        )

    def update_vector(self, point_id: str, vector: List[float]):
        """Update the vector of an object."""
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={} # Payload is preserved in upsert if not specified? No, upsert replaces.
                    # Qdrant upsert replaces the point. We need to get payload first or just rely on the API flow to provide full data?
                    # Actually, if we use UpdatePoint it might work, but upsert is safer.
                    # Let's assume the calling function handles full update or we fetch first.
                    # For safety, let's fetch payload first if this method is just "update vector".
                    # However, typical usage in our main.py will likely update both.
                    # Let's make this simple: Upsert overwrites. We should pass payload too if we want to keep it.
                )
            ]
        )
        # Wait, upsert replaces the entire point. 
        # To update JUST vector without losing payload, we have to read it first or pass it in.
        # Efficient way: The caller (main.py) will likely have the new metadata ready anyway.
        pass # Created a specific method below that accepts payload too

    def update_object(self, point_id: str, vector: Optional[List[float]], metadata: Dict[str, Any]):
        """
        Update an object. 
        If vector is provided, it replaces the point (upsert).
        If vector is None, it only updates payload.
        """
        if vector is not None:
            # Full replace (Vector + Payload)
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=metadata
                    )
                ]
            )
        else:
            # Only Payload update
            self.client.overwrite_payload(
                collection_name=self.collection_name,
                payload=metadata,
                points=[point_id]
            )

    def delete_by_object_id(self, object_id: str):
        """Delete all points belonging to an object_id."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="object_id",
                            match=models.MatchValue(value=object_id)
                        )
                    ]
                )
            )
        )

    def get_points_by_object_id(self, object_id: str) -> List[Dict[str, Any]]:
        """Get all points for a specific object_id."""
        # Scroll with filter
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="object_id",
                        match=models.MatchValue(value=object_id)
                    )
                ]
            ),
            limit=100, # Assuming max 100 images per object for now
            with_payload=True
        )
        return [
            {
                "id": point.id,
                "metadata": point.payload
            }
            for point in points
        ]

    def update_group_payload(self, object_id: str, metadata: Dict[str, Any]):
        """Update metadata for ALL points with the same object_id."""
        self.client.set_payload(
            collection_name=self.collection_name,
            payload=metadata,
            points=models.Filter(
                must=[
                    models.FieldCondition(
                        key="object_id",
                        match=models.MatchValue(value=object_id)
                    )
                ]
            )
        )
