#!/usr/bin/env python3
"""
Dataset Creator - Embeddings'lerden en çeşitli resimleri seçerek golden dataset oluşturur.
Diversity-based sampling ile benzer resimleri eler.

Spot instance desteği: Checkpoint/resume sistemi ile kesintilerde kaldığı yerden devam eder.
"""

import os
import json
import argparse
import shutil
import signal
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MiniBatchKMeans


# Global state for graceful shutdown
_shutdown_requested = False
CHECKPOINT_FILE = "creator_checkpoint.json"


def signal_handler(signum, frame):
    """Graceful shutdown için signal handler."""
    global _shutdown_requested
    print(f"\n⚠️  Sinyal alındı ({signum}), güvenli kapatma başlatılıyor...")
    _shutdown_requested = True


def setup_signal_handlers():
    """SIGTERM ve SIGINT için handler'ları ayarla."""
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def save_checkpoint(output_dir: Path, state: dict):
    """Checkpoint kaydet."""
    checkpoint_path = output_dir / CHECKPOINT_FILE
    temp_path = output_dir / f"{CHECKPOINT_FILE}.tmp"
    
    state['last_updated'] = datetime.now().isoformat()
    
    with open(temp_path, 'w') as f:
        json.dump(state, f, indent=2)
    temp_path.rename(checkpoint_path)
    
    print(f"💾 Checkpoint kaydedildi: {state.get('stage', 'unknown')}")


def load_checkpoint(output_dir: Path) -> Optional[dict]:
    """Varsa checkpoint'i yükle."""
    checkpoint_path = output_dir / CHECKPOINT_FILE
    
    if not checkpoint_path.exists():
        return None
    
    try:
        with open(checkpoint_path) as f:
            state = json.load(f)
        print(f"📂 Checkpoint bulundu: {state.get('last_updated', 'unknown')}")
        print(f"   Stage: {state.get('stage', 'unknown')}")
        return state
    except Exception as e:
        print(f"⚠️  Checkpoint okunamadı: {e}")
        return None


def load_embeddings(embeddings_dir: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    """Tüm embeddings ve metadata'yı yükle."""
    
    summary_file = embeddings_dir / "summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary dosyası bulunamadı: {summary_file}")
    
    with open(summary_file) as f:
        summary = json.load(f)
    
    all_embeddings = []
    all_metadata = []
    
    print(f"Toplam {len(summary['folders'])} klasörden embedding yükleniyor...")
    
    for folder_name, info in tqdm(summary['folders'].items()):
        emb_file = Path(info['embeddings_file'])
        meta_file = Path(info['metadata_file'])
        
        if not emb_file.exists() or not meta_file.exists():
            print(f"Dosya bulunamadı, atlanıyor: {folder_name}")
            continue
        
        embeddings = np.load(emb_file)
        metadata = pd.read_parquet(meta_file)
        
        all_embeddings.append(embeddings)
        all_metadata.append(metadata)
    
    combined_embeddings = np.vstack(all_embeddings)
    combined_metadata = pd.concat(all_metadata, ignore_index=True)
    
    print(f"Toplam yüklenen: {len(combined_metadata)} resim")
    print(f"Embedding boyutu: {combined_embeddings.shape}")
    
    return combined_embeddings, combined_metadata


def select_diverse_samples_kmeans(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    target_count: int,
    n_clusters: Optional[int] = None
) -> pd.DataFrame:
    """
    K-Means clustering ile her cluster'dan en merkeze yakın örnekleri seç.
    Bu yöntem hızlı ve etkili diversity sampling sağlar.
    """
    
    n_samples = len(embeddings)
    
    if target_count >= n_samples:
        print(f"Hedef ({target_count}) >= mevcut ({n_samples}), tüm örnekler seçiliyor.")
        return metadata
    
    # Cluster sayısını belirle
    if n_clusters is None:
        # Hedef sayının 1.5 katı cluster oluştur, sonra her cluster'dan seç
        n_clusters = min(target_count * 2, n_samples // 2)
    
    n_clusters = max(n_clusters, target_count)
    
    print(f"K-Means clustering başlıyor ({n_clusters} cluster)...")
    
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=1024,
        n_init=3
    )
    
    cluster_labels = kmeans.fit_predict(embeddings)
    cluster_centers = kmeans.cluster_centers_
    
    # Her cluster için merkeze en yakın örnekleri bul
    print("Her cluster'dan en iyi örnekler seçiliyor...")
    
    selected_indices = []
    samples_per_cluster = max(1, target_count // n_clusters)
    remaining = target_count
    
    # Cluster boyutlarına göre sırala (büyük cluster'lara daha fazla hak)
    cluster_sizes = np.bincount(cluster_labels, minlength=n_clusters)
    cluster_order = np.argsort(-cluster_sizes)  # Büyükten küçüğe
    
    for cluster_id in tqdm(cluster_order, desc="Cluster processing"):
        if remaining <= 0:
            break
            
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) == 0:
            continue
        
        # Cluster merkezine uzaklıkları hesapla
        cluster_embeddings = embeddings[cluster_indices]
        center = cluster_centers[cluster_id].reshape(1, -1)
        
        # Cosine similarity (1'e yakın = benzer)
        similarities = cosine_similarity(cluster_embeddings, center).flatten()
        
        # En benzer olanları seç (merkeze en yakın)
        n_select = min(samples_per_cluster, len(cluster_indices), remaining)
        top_indices = np.argsort(-similarities)[:n_select]
        
        selected_indices.extend(cluster_indices[top_indices])
        remaining -= n_select
    
    # Eğer hala eksik varsa, rastgele ekle
    if remaining > 0:
        all_indices = set(range(len(embeddings)))
        selected_set = set(selected_indices)
        remaining_indices = list(all_indices - selected_set)
        
        if remaining_indices:
            np.random.seed(42)
            extra = np.random.choice(remaining_indices, min(remaining, len(remaining_indices)), replace=False)
            selected_indices.extend(extra)
    
    selected_indices = list(set(selected_indices))[:target_count]
    
    print(f"Seçilen örnek sayısı: {len(selected_indices)}")
    
    return metadata.iloc[selected_indices].reset_index(drop=True)


def select_diverse_samples_greedy(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    target_count: int,
    batch_size: int = 1000,
    similarity_threshold: Optional[float] = None,
    min_count: Optional[int] = None
) -> pd.DataFrame:
    """
    Greedy diversity sampling - her adımda mevcut seçime en uzak örneği ekle.
    Daha yavaş ama daha iyi diversity. Büyük datasetler için batch processing.
    
    Args:
        embeddings: Resim embedding'leri
        metadata: Resim metadata'sı
        target_count: Hedef resim sayısı (max)
        batch_size: Batch boyutu
        similarity_threshold: Eğer bir resim mevcut seçimdeki herhangi bir resimle
                             bu threshold'dan daha benzer ise (similarity > threshold),
                             reddedilir. Örn: 0.95 = %95 benzerlik üstü reddedilir.
                             None ise threshold kontrolü yapılmaz.
        min_count: Minimum resim sayısı. Bu sayıya ulaşılamazsa uyarı verilir.
    
    Returns:
        Seçilen resimlerin metadata'sı
    """
    
    n_samples = len(embeddings)
    
    if target_count >= n_samples:
        print(f"Hedef ({target_count}) >= mevcut ({n_samples}), tüm örnekler seçiliyor.")
        return metadata
    
    print(f"Greedy diversity sampling başlıyor...")
    if similarity_threshold is not None:
        print(f"   Similarity threshold: {similarity_threshold} (>{similarity_threshold} benzerlik reddedilir)")
    if min_count is not None:
        print(f"   Minimum hedef: {min_count} resim")
    print(f"Bu işlem büyük datasetlerde uzun sürebilir.")
    
    selected_indices = []
    remaining_indices = list(range(n_samples))
    rejected_count = 0
    
    # İlk örneği rastgele seç
    np.random.seed(42)
    first_idx = np.random.choice(remaining_indices)
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    
    # Her örnek için seçilmiş örneklere olan max similarity
    max_similarities = np.zeros(n_samples)
    
    # İlk örneğe olan similarity'leri hesapla
    first_embedding = embeddings[first_idx].reshape(1, -1)
    initial_sims = cosine_similarity(embeddings, first_embedding).flatten()
    max_similarities = initial_sims.copy()
    
    pbar = tqdm(total=target_count - 1, desc="Greedy selection")
    
    while len(selected_indices) < target_count and remaining_indices:
        # En düşük max_similarity'ye sahip örneği bul (en farklı olan)
        remaining_max_sims = [(idx, max_similarities[idx]) for idx in remaining_indices]
        
        # En düşük similarity'den başlayarak sırala
        remaining_max_sims.sort(key=lambda x: x[1])
        
        found_valid = False
        
        for candidate_idx, candidate_max_sim in remaining_max_sims:
            # Threshold kontrolü
            if similarity_threshold is not None and candidate_max_sim > similarity_threshold:
                # Bu ve sonraki tüm adaylar threshold'u geçiyor (sıralı olduğu için)
                # Ama belki daha düşük similarity'li başka adaylar var, devam et
                rejected_count += 1
                remaining_indices.remove(candidate_idx)
                continue
            
            # Bu aday kabul edildi
            selected_indices.append(candidate_idx)
            remaining_indices.remove(candidate_idx)
            found_valid = True
            pbar.update(1)
            
            # Yeni eklenen örneğe olan similarity'leri güncelle
            new_embedding = embeddings[candidate_idx].reshape(1, -1)
            
            # Batch halinde similarity hesapla
            for batch_start in range(0, len(remaining_indices), batch_size):
                batch_indices = remaining_indices[batch_start:batch_start + batch_size]
                batch_embeddings = embeddings[batch_indices]
                
                similarities = cosine_similarity(batch_embeddings, new_embedding).flatten()
                
                for j, idx in enumerate(batch_indices):
                    max_similarities[idx] = max(max_similarities[idx], similarities[j])
            
            break
        
        if not found_valid:
            # Tüm kalan adaylar threshold'u geçiyor
            print(f"\n⚠️  Kalan tüm adaylar similarity threshold'unu geçiyor. Durduruluyor.")
            break
    
    pbar.close()
    
    print(f"\nSeçilen örnek sayısı: {len(selected_indices)}")
    print(f"Reddedilen (threshold): {rejected_count}")
    
    # Minimum kontrol
    if min_count is not None and len(selected_indices) < min_count:
        print(f"⚠️  UYARI: Minimum hedef ({min_count}) karşılanamadı! Sadece {len(selected_indices)} resim seçildi.")
        print(f"   Threshold'u düşürmeyi veya daha fazla kaynak resim eklemeyi düşünün.")
    
    return metadata.iloc[selected_indices].reset_index(drop=True)


def balance_folders(
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    target_count: int,
    min_per_folder: int = 100,
    max_per_folder: int = 2000
) -> pd.DataFrame:
    """
    Klasörler arası dengeyi koru. Her klasörden minimum sayıda al,
    sonra kalan kotayı diversity'ye göre dağıt.
    """
    
    folders = metadata['folder'].unique()
    n_folders = len(folders)
    
    print(f"Klasör sayısı: {n_folders}")
    
    # Her klasöre minimum kota
    base_per_folder = min(min_per_folder, target_count // n_folders)
    remaining_quota = target_count - (base_per_folder * n_folders)
    
    selected_dfs = []
    
    for folder in tqdm(folders, desc="Klasörler işleniyor"):
        folder_mask = metadata['folder'] == folder
        folder_indices = np.where(folder_mask)[0]
        folder_embeddings = embeddings[folder_indices]
        folder_metadata = metadata[folder_mask].copy()
        
        # Bu klasör için hedef sayı
        folder_count = len(folder_metadata)
        folder_target = min(
            max(base_per_folder, min_per_folder),
            max_per_folder,
            folder_count
        )
        
        if folder_count <= folder_target:
            selected_dfs.append(folder_metadata)
        else:
            # K-Means ile en çeşitli örnekleri seç
            selected = select_diverse_samples_kmeans(
                folder_embeddings,
                folder_metadata.reset_index(drop=True),
                folder_target,
                n_clusters=min(folder_target * 2, folder_count // 2)
            )
            selected_dfs.append(selected)
    
    result = pd.concat(selected_dfs, ignore_index=True)
    
    # Eğer hedefin altındaysak, en az seçilmiş klasörlerden ekle
    if len(result) < target_count:
        print(f"Hedefin altında: {len(result)} < {target_count}")
        # Şimdilik bu durumu kabul et
    
    # Eğer hedefin üstündeysen, global diversity ile kırp
    if len(result) > target_count:
        print(f"Hedefin üstünde: {len(result)} > {target_count}, kırpılıyor...")
        result_indices = result.index.tolist()
        result_embeddings = embeddings[folder_mask]  # Bu hatalı, düzeltilmeli
        # Basitçe rastgele seç
        np.random.seed(42)
        keep_indices = np.random.choice(len(result), target_count, replace=False)
        result = result.iloc[keep_indices].reset_index(drop=True)
    
    return result


def copy_selected_images(
    selected_metadata: pd.DataFrame,
    output_dir: Path,
    organize_by_folder: bool = True,
    checkpoint_dir: Optional[Path] = None
):
    """Seçilen resimleri çıktı klasörüne kopyala. Resume destekli."""
    global _shutdown_requested
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Daha önce kopyalanmış dosyaları bul
    copied_set = set()
    copied_file = checkpoint_dir / "copied_images.txt" if checkpoint_dir else None
    
    if copied_file and copied_file.exists():
        with open(copied_file) as f:
            copied_set = set(line.strip() for line in f)
        print(f"📂 Daha önce kopyalanan: {len(copied_set)} resim")
    
    print(f"\n{len(selected_metadata)} resim kopyalanıyor...")
    
    copied = len(copied_set)
    failed = 0
    skipped = 0
    
    # Append modunda aç
    copied_file_handle = open(copied_file, 'a') if copied_file else None
    
    try:
        for _, row in tqdm(selected_metadata.iterrows(), total=len(selected_metadata)):
            # Shutdown kontrolü
            if _shutdown_requested:
                print("⚠️  Kopyalama durduruluyor...")
                break
            
            src_path = Path(row['image_path'])
            src_str = str(src_path)
            
            # Zaten kopyalandıysa atla
            if src_str in copied_set:
                skipped += 1
                continue
            
            if not src_path.exists():
                failed += 1
                continue
            
            if organize_by_folder:
                dest_folder = output_dir / row['folder']
                dest_folder.mkdir(exist_ok=True)
                dest_path = dest_folder / src_path.name
            else:
                dest_path = output_dir / f"{row['folder']}_{src_path.name}"
            
            # Aynı isimde dosya varsa, sayı ekle
            if dest_path.exists():
                stem = dest_path.stem
                suffix = dest_path.suffix
                counter = 1
                while dest_path.exists():
                    dest_path = dest_path.parent / f"{stem}_{counter}{suffix}"
                    counter += 1
            
            try:
                shutil.copy2(src_path, dest_path)
                copied += 1
                
                # Kopyalanan dosyayı kaydet
                if copied_file_handle:
                    copied_file_handle.write(f"{src_str}\n")
                    copied_file_handle.flush()
                    
            except Exception as e:
                print(f"Kopyalama hatası: {src_path} -> {e}")
                failed += 1
    finally:
        if copied_file_handle:
            copied_file_handle.close()
    
    print(f"Kopyalanan: {copied}, Atlanan (önceden): {skipped}, Başarısız: {failed}")
    return not _shutdown_requested  # True if completed


def create_dataset(
    embeddings_dir: Path,
    output_dir: Path,
    target_count: int = 30000,
    method: str = "kmeans",
    copy_images: bool = True,
    organize_by_folder: bool = True,
    resume: bool = True,
    similarity_threshold: Optional[float] = None,
    min_count: Optional[int] = None
):
    """Ana dataset oluşturma fonksiyonu. Resume destekli."""
    global _shutdown_requested
    
    # Signal handler'ları ayarla
    setup_signal_handlers()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Checkpoint kontrolü
    checkpoint = load_checkpoint(output_dir) if resume else None
    current_stage = checkpoint.get('stage', 'start') if checkpoint else 'start'
    
    selection_file = output_dir / "selected_images.parquet"
    
    # Stage: Selection
    if current_stage in ['start', 'selection']:
        if selection_file.exists() and resume:
            print(f"📂 Önceki seçim yükleniyor: {selection_file}")
            selected = pd.read_parquet(selection_file)
            print(f"   Yüklenen: {len(selected)} resim")
        else:
            # Embeddings yükle
            embeddings, metadata = load_embeddings(embeddings_dir)
            
            print(f"\nHedef dataset boyutu: {target_count}")
            print(f"Seçim yöntemi: {method}")
            
            # Diversity-based selection
            if method == "kmeans":
                selected = select_diverse_samples_kmeans(
                    embeddings, metadata, target_count
                )
            elif method == "greedy":
                selected = select_diverse_samples_greedy(
                    embeddings, metadata, target_count,
                    similarity_threshold=similarity_threshold,
                    min_count=min_count
                )
            elif method == "balanced":
                selected = balance_folders(
                    metadata, embeddings, target_count
                )
            else:
                raise ValueError(f"Bilinmeyen yöntem: {method}")
            
            # Seçimi kaydet
            selected.to_parquet(selection_file, index=False)
            print(f"\n✅ Seçim kaydedildi: {selection_file}")
            
            # CSV olarak da kaydet
            csv_file = output_dir / "selected_images.csv"
            selected.to_csv(csv_file, index=False)
            print(f"✅ CSV kaydedildi: {csv_file}")
        
        # Checkpoint güncelle
        save_checkpoint(output_dir, {
            'stage': 'copy' if copy_images else 'stats',
            'method': method,
            'target_count': target_count,
            'selected_count': len(selected)
        })
        
        current_stage = 'copy' if copy_images else 'stats'
    else:
        # Selection zaten tamamlanmış, yükle
        selected = pd.read_parquet(selection_file)
    
    if _shutdown_requested:
        print("\n⚠️  İşlem durduruldu.")
        return
    
    # Stage: Copy
    if current_stage == 'copy' and copy_images:
        images_dir = output_dir / "images"
        copy_completed = copy_selected_images(
            selected, images_dir, organize_by_folder,
            checkpoint_dir=output_dir
        )
        
        if copy_completed:
            save_checkpoint(output_dir, {
                'stage': 'stats',
                'method': method,
                'target_count': target_count,
                'selected_count': len(selected)
            })
            current_stage = 'stats'
        else:
            print("\n⚠️  Kopyalama yarıda kesildi. Devam etmek için tekrar çalıştırın.")
            return
    
    if _shutdown_requested:
        print("\n⚠️  İşlem durduruldu.")
        return
    
    # Stage: Stats
    if current_stage == 'stats':
        # İstatistikler
        print(f"\n{'='*50}")
        print("DATASET İSTATİSTİKLERİ")
        print(f"{'='*50}")
        print(f"Toplam seçilen resim: {len(selected)}")
        print(f"\nKlasör dağılımı:")
        folder_counts = selected['folder'].value_counts()
        for folder, count in folder_counts.items():
            print(f"  {folder}: {count}")
        
        # Summary kaydet
        summary = {
            "total_selected": len(selected),
            "method": method,
            "target_count": target_count,
            "folder_distribution": folder_counts.to_dict(),
            "source_embeddings_dir": str(embeddings_dir),
            "completed_at": datetime.now().isoformat()
        }
        
        with open(output_dir / "dataset_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Checkpoint temizle (başarıyla tamamlandı)
        checkpoint_path = output_dir / CHECKPOINT_FILE
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        
        copied_file = output_dir / "copied_images.txt"
        if copied_file.exists():
            copied_file.unlink()
        
        print(f"\n{'='*50}")
        print("✅ TAMAMLANDI!")
        print(f"Dataset klasörü: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Dataset Creator - Golden dataset oluştur')
    parser.add_argument('--embeddings', '-e', type=str, required=True,
                        help='Embeddings klasörü (embedder.py çıktısı)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Çıktı klasörü')
    parser.add_argument('--target', '-t', type=int, default=30000,
                        help='Hedef resim sayısı (default: 30000)')
    parser.add_argument('--method', '-m', type=str, default='kmeans',
                        choices=['kmeans', 'greedy', 'balanced'],
                        help='Seçim yöntemi (default: kmeans)')
    parser.add_argument('--no-copy', action='store_true',
                        help='Resimleri kopyalama, sadece liste oluştur')
    parser.add_argument('--flat', action='store_true',
                        help='Klasör yapısı olmadan düz kopyala')
    parser.add_argument('--no-resume', action='store_true',
                        help='Checkpoint kullanma, sıfırdan başla')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Similarity threshold (0-1). Örn: 0.95 = %%95 üstü benzerlik reddedilir. Sadece greedy için.')
    parser.add_argument('--min-count', type=int, default=None,
                        help='Minimum resim sayısı. Bu sayıya ulaşılamazsa uyarı verilir.')
    
    args = parser.parse_args()
    
    print(f"🚀 Dataset Creator başlatılıyor...")
    print(f"   Resume: {'Hayır' if args.no_resume else 'Evet'}")
    
    create_dataset(
        embeddings_dir=Path(args.embeddings),
        output_dir=Path(args.output),
        target_count=args.target,
        method=args.method,
        copy_images=not args.no_copy,
        organize_by_folder=not args.flat,
        resume=not args.no_resume,
        similarity_threshold=args.threshold,
        min_count=args.min_count
    )


if __name__ == "__main__":
    main()
