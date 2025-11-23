#!/usr/bin/env python3
import os
import cv2
from photosynth.db import PhotoSynthDB
from photosynth.pipeline.detector import Detector
from tqdm import tqdm

def main():
    print("🖼️ Generating Cluster Thumbnails...")
    db = PhotoSynthDB()
    detector = Detector(enable_yolo=False) # Light mode
    
    # 1. Get all clusters that need thumbnails
    print("   Fetching clusters from DB...")
    # Get list of (cluster_id)
    clusters = db.get_connection().execute("SELECT cluster_id FROM people").fetchall()
    
    if not clusters:
        print("❌ No clusters found. Run cluster_faces.py first.")
        return

    print(f"   Found {len(clusters)} clusters. Generating thumbnails...")

    generated_count = 0
    
    for row in tqdm(clusters):
        cluster_id = row[0]
        
        # 2. Find up to 5 source images for this person
        query = """
            SELECT m.file_path 
            FROM faces f 
            JOIN media_files m ON f.file_hash = m.file_hash 
            WHERE f.cluster_id = ? 
            LIMIT 5
        """
        files = db.get_connection().execute(query, (cluster_id,)).fetchall()
        
        for f_row in files:
            path = f_row[0]
            path = detector._heal_path(path)
            
            try:
                if not os.path.exists(path): continue
                
                # 3. Re-detect and Save Crop
                # We manually trigger the save logic here
                img = cv2.imread(path)
                if img is None: continue
                
                faces = detector.face_app.get(img)
                
                # Save ALL faces in this image (easiest way to ensure we get the right one)
                # The backend will filter them anyway.
                detector._save_face_crops(faces, img, path)
                generated_count += 1
                
            except Exception as e:
                print(f"Error processing {path}: {e}")

    print(f"✅ Generated {generated_count} thumbnails.")
    print("👉 Refresh the UI!")

if __name__ == "__main__":
    main()