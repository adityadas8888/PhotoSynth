#!/usr/bin/env python3
import os
from pathlib import Path
from tqdm import tqdm
from photosynth.db import PhotoSynthDB
from photosynth.pipeline.detector import Detector
from photosynth.utils.hashing import calculate_content_hash

# Config
NAS_PATH = os.path.expanduser("~/personal/nas/video/TEST")
EXTENSIONS = ['.jpg', '.jpeg', '.png', '.arw']

def main():
    print("🚀 Starting Visual-Hash Face Harvest...")
    db = PhotoSynthDB()
    
    print("   Loading DB index...")
    conn = db.get_connection()
    known_hashes = set(row[0] for row in conn.execute("SELECT file_hash FROM faces").fetchall())
    conn.close()
    
    detector = Detector(enable_yolo=False)
    
    files = []
    for ext in EXTENSIONS:
        files.extend(Path(NAS_PATH).rglob(f"*{ext}"))
        files.extend(Path(NAS_PATH).rglob(f"*{ext.upper()}"))
    
    print(f"📂 Checking {len(files)} files...")

    new_count = 0
    for p in tqdm(files):
        path_str = str(p)
        if "@eaDir" in path_str: continue
        
        # VISUAL HASH CHECK
        f_hash = calculate_content_hash(path_str)
        if not f_hash or f_hash in known_hashes: continue

        try:
            result = detector._process_image(path_str)
            embeddings = result.get('faces', [])
            if embeddings:
                db.register_file(f_hash, path_str)
                for emb in embeddings:
                    import numpy as np
                    db.add_face(f_hash, np.array(emb, dtype=np.float32))
                new_count += 1
        except: pass

    print(f"✅ Harvest Complete. Scanned {new_count} new files.")

if __name__ == "__main__":
    main()