#!/usr/bin/env python3
import os
from pathlib import Path
from tqdm import tqdm
from photosynth.db import PhotoSynthDB
from photosynth.pipeline.detector import Detector
from photosynth.utils.hashing import calculate_content_hash

# Config
NAS_PATH = os.path.expanduser("~/personal/nas/photo")
# Only harvest faces from Images. Videos are too blurry/numerous for the cluster DB.
EXTENSIONS = ['.jpg', '.jpeg', '.png', '.arw', '.heic']

def main():
    print("🚀 Starting Smart Face Harvest...")
    db = PhotoSynthDB()
    
    # 1. Load Cache (The Speed Trick)
    print("   Loading DB index...")
    conn = db.get_connection()
    
    # Get all known paths (Fastest check)
    known_paths = set(row[0] for row in conn.execute("SELECT file_path FROM media_files").fetchall())
    
    # Get all known visual hashes (Check for moved files)
    known_hashes = set(row[0] for row in conn.execute("SELECT file_hash FROM faces").fetchall())
    
    conn.close()
    print(f"   Loaded {len(known_paths)} paths and {len(known_hashes)} hashes.")
    
    # Initialize Detector in Light Mode (No YOLO, just Faces)
    detector = Detector(enable_yolo=False)
    
    # 2. Find Files
    print("   Listing files on NAS...")
    files = []
    for ext in EXTENSIONS:
        files.extend(Path(NAS_PATH).rglob(f"*{ext}"))
        files.extend(Path(NAS_PATH).rglob(f"*{ext.upper()}"))
    
    print(f"📂 Checking {len(files)} files...")

    new_count = 0
    path_skipped = 0
    hash_skipped = 0
    
    for p in tqdm(files):
        path_str = str(p)
        if "@eaDir" in path_str: continue
        
        # --- LEVEL 1: PATH CHECK (Instant) ---
        if path_str in known_paths:
            path_skipped += 1
            continue

        # --- LEVEL 2: HASH CHECK (Slow IO, but no GPU) ---
        # Only calc hash if path is unknown
        f_hash = calculate_content_hash(path_str)
        if not f_hash: continue
            
        if f_hash in known_hashes:
            # It's a renamed/moved file. Register the new path, but don't re-scan faces.
            db.register_file(f_hash, path_str)
            hash_skipped += 1
            continue

        # --- LEVEL 3: GPU PROCESS ---
        try:
            # Extract embeddings
            result = detector._process_image(path_str)
            embeddings = result.get('faces', [])
            
            # Always register file (even if no faces found, so we don't scan it again)
            db.register_file(f_hash, path_str)
            
            if embeddings:
                for emb in embeddings:
                    import numpy as np
                    db.add_face(f_hash, np.array(emb, dtype=np.float32))
                new_count += 1
                
        except Exception as e:
            # print(f"Error: {e}")
            pass

    print("-" * 30)
    print(f"✅ Harvest Complete.")
    print(f"⏩ Skipped (Path Match): {path_skipped}")
    print(f"🔍 Skipped (Hash Match): {hash_skipped}")
    print(f"🆕 Scanned: {new_count}")
    print("-" * 30)
    print("👉 Now run: uv run scripts/cluster_faces.py")

if __name__ == "__main__":
    main()