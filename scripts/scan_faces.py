#!/usr/bin/env python3
import os
import hashlib
from pathlib import Path
from tqdm import tqdm
from photosynth.db import PhotoSynthDB
from photosynth.pipeline.detector import Detector

# Config
NAS_PATH = os.path.expanduser("~/personal/nas/video/TEST")
EXTENSIONS = ['.jpg', '.jpeg', '.png', '.arw']

def get_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        hasher.update(f.read(65536))
    return hasher.hexdigest()

def main():
    print("🚀 Starting Monthly Face Harvest...")
    
    # Initialize Detector WITHOUT YOLO (Save VRAM/Time)
    detector = Detector(enable_yolo=False)
    db = PhotoSynthDB()
    
    files = []
    for ext in EXTENSIONS:
        files.extend(Path(NAS_PATH).rglob(f"*{ext}"))
    
    print(f"📂 Scanning {len(files)} files for faces...")

    count = 0
    for p in tqdm(files):
        path_str = str(p)
        if "@eaDir" in path_str: continue
        
        try:
            # Only scan if not already processed/clustered? 
            # For now, we scan everything to ensure completeness.
            
            result = detector._process_image(path_str)
            embeddings = result.get('faces', [])
            
            if embeddings:
                f_hash = get_hash(path_str)
                db.register_file(f_hash, path_str)
                
                for emb in embeddings:
                    # Convert list back to numpy
                    import numpy as np
                    db.add_face(f_hash, np.array(emb, dtype=np.float32))
                    count += 1
                    
        except Exception as e:
            # print(f"Skipping {path_str}: {e}")
            pass

    print(f"✅ Harvest Complete. Found {count} faces.")
    print("👉 Now run: uv run scripts/cluster_faces.py")

if __name__ == "__main__":
    main()