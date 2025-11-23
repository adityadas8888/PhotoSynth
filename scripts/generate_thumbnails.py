#!/usr/bin/env python3
import os
import cv2
from photosynth.db import PhotoSynthDB
from photosynth.pipeline.detector import Detector
from tqdm import tqdm

def main():
    print("🖼️ Generating Cluster Thumbnails...")
    db = PhotoSynthDB()
    detector = Detector(enable_yolo=False)
    
    clusters = db.get_connection().execute("SELECT cluster_id FROM people").fetchall()
    print(f"   Found {len(clusters)} clusters.")

    for row in tqdm(clusters):
        cluster_id = row[0]
        
        query = """
            SELECT m.file_path 
            FROM faces f 
            JOIN media_files m ON f.file_hash = m.file_hash 
            WHERE f.cluster_id = ? 
            LIMIT 5
        """
        files = db.get_connection().execute(query, (cluster_id,)).fetchall()
        
        for f_row in files:
            path = detector._heal_path(f_row[0])
            if not os.path.exists(path): continue
            
            try:
                img = cv2.imread(path)
                if img is None: continue
                faces = detector.face_app.get(img)
                detector._save_face_crops(faces, img, path)
            except: pass

    print("✅ Thumbnails generated. Refresh UI!")

if __name__ == "__main__":
    main()