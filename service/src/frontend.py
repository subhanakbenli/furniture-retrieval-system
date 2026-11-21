import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io
import json
import uuid
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(layout="wide", page_title="YOLO+CLIP Object Search")

def draw_detections(image, detections):
    """Draws bounding boxes and labels on the image."""
    draw = ImageDraw.Draw(image)
    # Try to load a font, optional
    try:
        font = ImageFont.truetype("arial.ttf", 30) # Larger font
    except IOError:
        font = None # Use default

    for i, det in enumerate(detections):
        box = det['box'] # [x1, y1, x2, y2]
        # Label: "#1 class (conf)"
        label = f"#{i+1} {det['label']} ({det['confidence']:.2f})"
        
        # Draw Box
        draw.rectangle(box, outline="red", width=4)
        
        # Draw Label background
        if font:
            left, top, right, bottom = draw.textbbox((box[0], box[1]), label, font=font)
            text_loc = (box[0], box[1] - (bottom-top) - 5 if box[1] > (bottom-top)+5 else box[1])
            draw.rectangle((left, text_loc[1], right, text_loc[1] + (bottom-top)), fill="red")
            draw.text((left, text_loc[1]), label, fill="white", font=font)
        else:
            # Fallback simple
            text_loc = (box[0], box[1] - 15 if box[1] > 15 else box[1])
            draw.rectangle([text_loc, (text_loc[0] + 150, text_loc[1] + 15)], fill="red")
            draw.text(text_loc, label, fill="white")
    
    return image

def show_stored_image(img_url, **kwargs):
    """Fetch a stored product image through the API (server side), so it also
    works when the browser can't reach API_URL directly (e.g. inside Docker)."""
    try:
        resp = requests.get(f"{API_URL}{img_url}", timeout=10)
        resp.raise_for_status()
        st.image(Image.open(io.BytesIO(resp.content)), **kwargs)
    except Exception:
        st.text("Image unavailable")

st.title("🔎 Object Detection & Semantic Search")

tab_search, tab_store, tab_inventory, tab_visualize = st.tabs(["Search Objects", "Add to Database", "Inventory", "Visualize"])

if "inventory_offset" not in st.session_state:
    st.session_state.inventory_offset = None
if "edit_object" not in st.session_state:
    st.session_state.edit_object = None

def upload_image_to_api(endpoint, files, data=None):
    try:
        response = requests.post(f"{API_URL}/{endpoint}", files=files, data=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to backend at {API_URL}. Is it running?")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def render_add_object_flow(key_prefix, prefilled_metadata=None, on_success=None, on_cancel=None):
    """
    Reusable 2-stage flow: Upload -> Analyze -> Select/Edit Details -> Store.
    
    Args:
        key_prefix: Unique key for session state isolation.
        prefilled_metadata: Dict with existing meta (name, category, object_id) if adding variant.
                            If None, user must enter details.
        on_success: Callback function after successful store.
        on_cancel: Callback function to cancel.
    """
    
    # Local State Keys
    stage_key = f"{key_prefix}_stage"
    dets_key = f"{key_prefix}_detections"
    imgs_key = f"{key_prefix}_images"
    meta_key = f"{key_prefix}_metadata"
    
    # Initialize State
    if stage_key not in st.session_state:
        st.session_state[stage_key] = "upload"
    if dets_key not in st.session_state:
        st.session_state[dets_key] = {}
    if imgs_key not in st.session_state:
        st.session_state[imgs_key] = {}
        
    def reset_flow():
        st.session_state[stage_key] = "upload"
        st.session_state[dets_key] = {}
        st.session_state[imgs_key] = {}
        
    # --- STAGE 1: UPLOAD (& DATA ENTRY if new) ---
    if st.session_state[stage_key] == "upload":
        st.subheader("Step 1: Upload Images")
        
        # If adding variant, show context
        if prefilled_metadata:
            st.info(f"Adding new variant for **{prefilled_metadata.get('name')}**")
        
        uploaded_files = st.file_uploader("Choose Images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'], key=f"{key_prefix}_up")
        
        # Metadata Inputs (Only if creating NEW object)
        if not prefilled_metadata:
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Object Name (e.g. 'Red Chair')", key=f"{key_prefix}_name")
                category = st.selectbox("Category", ["Furniture", "Textile", "Lighting", "Decoration", "Appliance"], key=f"{key_prefix}_cat")
            with col2:
                price = st.text_input("Price (Optional)", key=f"{key_prefix}_price")
                description = st.text_area("Description", key=f"{key_prefix}_desc")
        
        col_act1, col_act2 = st.columns([1, 5])
        with col_act1:
            if st.button("Analyze Images", type="primary", key=f"{key_prefix}_analyze"):
                # Validation
                if not uploaded_files:
                    st.warning("Please upload at least one image.")
                    return
                
                final_meta = {}
                if prefilled_metadata:
                    final_meta = prefilled_metadata
                else:
                    if not name:
                        st.warning("Please enter an object name.")
                        return
                    final_meta = {
                        "name": name,
                        "category": category,
                        "price": price,
                        "description": description
                    }
                
                # Perform Analysis
                with st.spinner("Analyzing..."):
                    detections_map = {} 
                    images_map = {}
                    
                    for uf in uploaded_files:
                        img = Image.open(uf).convert('RGB')
                        images_map[uf.name] = img
                        
                        uf.seek(0)
                        files = {"file": (uf.name, uf, uf.type)}
                        result = upload_image_to_api("detect-and-search", files) # Just for detection
                        if result:
                            detections_map[uf.name] = result['results']
                    
                    st.session_state[dets_key] = detections_map
                    st.session_state[imgs_key] = images_map
                    st.session_state[meta_key] = final_meta
                    st.session_state[stage_key] = "select"
                    st.rerun()
                    
        with col_act2:
            if on_cancel and st.button("Cancel", key=f"{key_prefix}_cancel_1"):
                reset_flow()
                on_cancel()

    # --- STAGE 2: SELECT CROPS ---
    elif st.session_state[stage_key] == "select":
        st.subheader("Step 2: Verify & Select Objects")
        st.info("Uncheck any detections that are incorrect or not the target object.")
        
        meta = st.session_state[meta_key]
        
        with st.form(f"{key_prefix}_form"):
            has_images = False
            
            for filename, detections in st.session_state[dets_key].items():
                if not detections: continue
                has_images = True
                
                st.markdown(f"#### 📄 {filename}")
                img = st.session_state[imgs_key][filename]
                
                # Draw Numbered Detections
                annotated_img = draw_detections(img.copy(), detections)
                st.image(annotated_img, use_container_width=True)
                
                st.write("**Detected Objects (Select to Include):**")
                
                # Card Grid for Crops
                cols = st.columns(4)
                for i, det in enumerate(detections):
                    col = cols[i % 4]
                    with col:
                        # Crop Preview
                        box = det['box']
                        crop = img.crop(box)
                        st.image(crop, caption=f"Object #{i+1}", width=120)
                        
                        # Checkbox
                        lbl = f"#{i+1} {det['label']} ({det['confidence']:.2f})"
                        st.checkbox(lbl, value=True, key=f"{key_prefix}_chk_{filename}_{i}")
                
                st.divider()

            col_submit, col_cancel = st.columns([1, 1])
            with col_submit:
                if st.form_submit_button("Confirm & Save", type="primary"):
                    # Collect Crops
                    final_crops = []
                    for filename, detections in st.session_state[dets_key].items():
                        img = st.session_state[imgs_key][filename]
                        for i, det in enumerate(detections):
                            if st.session_state.get(f"{key_prefix}_chk_{filename}_{i}"):
                                box = det['box']
                                crop = img.crop(box)
                                buf = io.BytesIO()
                                crop.save(buf, format="JPEG")
                                buf.seek(0)
                                final_crops.append(
                                    ("files", (f"crop_{filename}_{i}.jpg", buf, "image/jpeg"))
                                )
                    
                    if final_crops:
                        # Submit Method depends on if it's new (POST /store) or update (PUT /objects/id)
                        # Actually PUT /objects/{id} adds variants too, but POST /store needs object_id if exists.
                        # Using POST /store for adding variants is also fine if we pass object_id.
                        
                        # Let's stick to POST /store which handles "Metadata has object_id -> append" logic
                        # provided the backend supports it. The backend code says:
                        # "Generate a shared object_id if not present" -> implies if present it uses it?
                        # Checking backend... Yes, in store_image: "if 'object_id' not in metadata_json: generate..."
                        # So POST /store is the correct unified endpoint.
                        
                        with st.spinner("Storing..."):
                            try:
                                resp = requests.post(
                                    f"{API_URL}/store",
                                    files=final_crops,
                                    data={"metadata": json.dumps(meta)}
                                )
                                if resp.status_code == 200:
                                    st.success("Saved successfully!")
                                    reset_flow()
                                    if on_success: on_success()
                                    # Optional delay for read
                                else:
                                    st.error(f"Failed: {resp.text}")
                            except Exception as e:
                                st.error(f"Error: {e}")
                    else:
                        st.warning("No objects selected!")
            
            with col_cancel:
                if on_cancel and st.form_submit_button("Cancel"):
                     reset_flow()
                     on_cancel()

# --- STORE TAB ---
with tab_store:
    st.header("Add Objects to Database (2-Stage)")
    
    def on_store_success():
        # Maybe show a button to add another?
        pass

    render_add_object_flow("store_tab")

# --- SEARCH TAB ---
with tab_search:
    st.header("Detect & Search")
    uploaded_file_search = st.file_uploader("Upload Scene Image", type=['png', 'jpg', 'jpeg'], key="search")
    
    if uploaded_file_search:
        col_src, col_res = st.columns([1, 1])
        
        # Display original first
        image_placeholder = col_src.empty()
        input_image = Image.open(uploaded_file_search).convert('RGB')
        image_placeholder.image(input_image, caption="Input Scene", use_container_width=True)
        
        with st.expander("Advanced Search Settings"):
            s_conf = st.slider("Detection Confidence", 0.0, 1.0, 0.25, 0.05, key="s_conf")
            s_sim = st.slider("Search Similarity Threshold", 0.0, 1.0, 0.60, 0.05, key="s_sim")
            
        if st.button("Detect & Find Matches"):
            with st.spinner("Detecting and Searching..."):
                uploaded_file_search.seek(0)
                files = {"file": ("scene.jpg", uploaded_file_search, "image/jpeg")}
                data = {
                    "conf_threshold": s_conf,
                    "search_threshold": s_sim,
                    "iof_threshold": 0.5 # Default for search view
                }
                
                result = upload_image_to_api("detect-and-search", files, data=data)
                
                if result:
                    st.success(f"Found {result['detections_count']} objects.")
                    
                    # Draw boxes
                    annotated_img = draw_detections(input_image.copy(), result['results'])
                    image_placeholder.image(annotated_img, caption="Detected Objects", use_container_width=True)
                    
                    with col_res:
                        st.subheader("Results by Object")
                        for idx, item in enumerate(result['results']):
                            with st.expander(f"Object {idx+1}: {item['label']} ({item['confidence']:.2f})", expanded=True):
                                # Display Box info
                                box = item['box']
                                st.write(f"**Bounding Box:** {box}")
                                
                                st.markdown("#### Top Matches:")
                                if not item['matches']:
                                    st.info("No matches found in database.")
                                else:
                                    for match in item['matches']:
                                        score = match['score']
                                        meta = match['metadata']
                                        
                                        # Show stored image if available
                                        img_url = meta.get("image_url")
                                        if img_url:
                                            show_stored_image(img_url, width=150)
                                            
                                        st.markdown(f"""
                                        - **{meta.get('name', 'Unknown')}** ({score:.3f})  
                                          _{meta.get('description', '')}_  
                                          *Category:* {meta.get('category', 'N/A')}
                                        """)

# --- INVENTORY TAB ---
with tab_inventory:
    st.header("Inventory Management")
    
    # Session state for inventory mode
    if "inv_mode" not in st.session_state:
        st.session_state.inv_mode = "list" # list, add_variant
    if "inv_target_object" not in st.session_state:
        st.session_state.inv_target_object = None
        
    def switch_to_list():
        st.session_state.inv_mode = "list"
        st.session_state.inv_target_object = None
        st.rerun()

    # --- MODE: ADD VARIANT ---
    if st.session_state.inv_mode == "add_variant":
        target = st.session_state.inv_target_object
        if target:
            if st.button("← Back to Inventory", type="secondary"):
                switch_to_list()
                
            render_add_object_flow(
                key_prefix=f"inv_add_{target['object_id']}",
                prefilled_metadata=target,
                on_success=switch_to_list,
                on_cancel=switch_to_list
            )
        else:
            st.error("Target object lost.")
            switch_to_list()

    # --- MODE: LIST ---
    else:
        # --- EDIT DIALOG ---
        @st.dialog("Edit Object")
        def edit_object_dialog(obj):
            st.write(f"Editing: {obj.get('name', 'Unknown')}")
            meta = obj['metadata']
            
            # Inputs
            new_name = st.text_input("Name", value=meta.get('name', ''), key=f"name_{obj['id']}")
            new_desc = st.text_area("Description", value=meta.get('description', ''), key=f"desc_{obj['id']}")
            
            # Category
            current_cat = meta.get('category', 'Other')
            cat_options = ["Furniture", "Textile", "Lighting", "Decoration", "Appliance"]
            try:
                cat_idx = cat_options.index(current_cat)
            except ValueError:
                cat_idx = 0
            new_cat = st.selectbox("Category", cat_options, index=cat_idx, key=f"cat_{obj['id']}")
            
            # (Image upload removed)
            
            if st.button("Save Changes", key=f"save_{obj['id']}"):
                new_meta = {
                    "name": new_name,
                    "description": new_desc,
                    "category": new_cat,
                    "object_id": meta.get("object_id", str(uuid.uuid4())), # Ensure object_id is passed
                    "image_url": meta.get("image_url") 
                }
                
                try:
                    # We need a PUT request
                    url = f"{API_URL}/objects/{obj['id']}"
                    data = {"metadata": json.dumps(new_meta)}
                    
                    resp = requests.put(url, data=data)
                    resp.raise_for_status()
                    st.success("Updated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to update: {e}")

        # Fetch Data
        params = {"limit": 100}
        if st.session_state.inventory_offset:
            params["offset"] = st.session_state.inventory_offset
            
        try:
            resp = requests.get(f"{API_URL}/objects", params=params)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                next_offset = data.get("next_offset")
                
                # Group by object_id
                grouped_items = {}
                for item in results:
                    meta = item['metadata']
                    # Fallback to point_id if object_id missing (legacy data)
                    obj_id = meta.get('object_id', item['id'])
                    
                    # Store cleanest metadata for the group (e.g. from first item)
                    # We also need the point ID for specific actions
                    if obj_id not in grouped_items:
                        grouped_items[obj_id] = []
                    grouped_items[obj_id].append(item)
                
                # Display Groups
                for obj_id, variants in grouped_items.items():
                    main_item = variants[0]
                    meta = main_item['metadata']
                    
                    with st.container(border=True):
                        col_img, col_info, col_act = st.columns([2, 5, 2])
                        
                        with col_img:
                             img_url = meta.get("image_url")
                             if img_url:
                                 show_stored_image(img_url, use_container_width=True)
                                 if len(variants) > 1:
                                     st.caption(f"+ {len(variants)-1} other variants")
                             else:
                                 st.text("No Image")
                        
                        with col_info:
                            st.subheader(meta.get("name", "Unnamed"))
                            st.write(meta.get("description", ""))
                            st.caption(f"Object ID: {obj_id} | Cat: {meta.get('category')}")
                        
                        with col_act:
                            if st.button("➕ Add Variant", key=f"add_var_{obj_id}"):
                                # Prepare metadata for the add flow
                                target_meta = {
                                    "object_id": obj_id,
                                    "name": meta.get("name", ""),
                                    "category": meta.get("category", "")
                                }
                                st.session_state.inv_target_object = target_meta
                                st.session_state.inv_mode = "add_variant"
                                st.rerun()
                            
                            if st.button("Edit Metadata", key=f"edit_{obj_id}"):
                                edit_object_dialog(main_item)
                                
                            if st.button("Delete Object", key=f"del_obj_{obj_id}", type="primary"):
                                try:
                                    for v in variants:
                                        requests.delete(f"{API_URL}/objects/{v['id']}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(str(e))
                    
                    # Variants Expander
                    with st.expander(f"View Inventory Variants ({len(variants)})"):
                        cols = st.columns(5)
                        for idx, v in enumerate(variants):
                            col = cols[idx % 5]
                            with col:
                                v_img = v['metadata'].get("image_url")
                                if v_img:
                                    show_stored_image(v_img, width=100)
                                
                                if st.button("🗑️", key=f"del_var_{v['id']}"):
                                    try:
                                        requests.delete(f"{API_URL}/objects/{v['id']}")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(str(e))
                                    
                # Pagination
                col_prev, col_next = st.columns(2)
                with col_next:
                    if next_offset:
                        if st.button("Load More"):
                            st.session_state.inventory_offset = next_offset
                            st.rerun()
                with col_prev:
                    if st.session_state.inventory_offset:
                        if st.button("Reset / First Page"):
                             st.session_state.inventory_offset = None
                             st.rerun()
                             
        except Exception as e:
            st.error(f"Failed to load inventory: {e}")

# --- VISUALIZE TAB ---
with tab_visualize:
    st.header("Post-Processing Visualizer")
    st.write("Tune detection parameters and see how the post-processing logic (Merge/Conflict Resolution) affects the results.")
    
    col_upload, col_controls = st.columns([1, 1])
    
    with col_upload:
        viz_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="viz_up")
        
    with col_controls:
        conf_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05, help="Minimum confidence for YOLO detections.")
        iof_thresh = st.slider("IoF Threshold (Merger)", 0.0, 1.0, 0.5, 0.05, help="Intersect over Foreground threshold for merging same-class boxes.")
        
    if viz_file:
        if st.button("Process & Visualize"):
            with st.spinner("Processing..."):
                try:
                    viz_file.seek(0)
                    files = {"file": ("viz.jpg", viz_file, "image/jpeg")}
                    data = {"conf_threshold": conf_thresh, "iof_threshold": iof_thresh}
                    
                    # Direct request since upload_image_to_api expects JSON return
                    resp = requests.post(f"{API_URL}/visualize", files=files, data=data)
                    
                    if resp.status_code == 200:
                        image = Image.open(io.BytesIO(resp.content))
                        st.image(image, caption="Processed Result", use_container_width=True)
                    else:
                        st.error(f"Error {resp.status_code}: {resp.text}")
                        
                except Exception as e:
                    st.error(f"Failed: {e}")
