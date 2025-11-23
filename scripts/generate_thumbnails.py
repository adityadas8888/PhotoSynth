#!/usr/bin/env python3
import os
import cv2
from photosynth.db import PhotoSynthDB
from photosynth.pipeline.detector import Detector
from photosynth.utils.paths import heal_path
from tqdm import tqdm

# Define output directory relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACES_DIR = os.path.join(BASE_DIR, "faces_crop")
os.makedirs(FACES_DIR, exist_ok=True)

def main():
    print(f"🖼️  Generating Cluster Thumbnails in: {FACES_DIR}")
    db = PhotoSynthDB()
    # Light mode is fine, we will handle saving manually
    detector = Detector(enable_yolo=False) 
    
    print("   Fetching clusters from DB...")
    clusters = db.get_connection().execute("SELECT cluster_id FROM people").fetchall()
    
    if not clusters:
        print("❌ No clusters found. Run cluster_faces.py first.")
        return

    print(f"   Found {len(clusters)} clusters. Generating thumbnails...")

    generated_count = 0
    
    for row in tqdm(clusters):
        cluster_id = row[0]
        
        # Get up to 5 images per person
        query = """
            SELECT m.file_path 
            FROM faces f 
            JOIN media_files m ON f.file_hash = m.file_hash 
            WHERE f.cluster_id = ? 
            LIMIT 5
        """
        files = db.get_connection().execute(query, (cluster_id,)).fetchall()
        
        for f_row in files:
            # 1. Path Healing
            path = heal_path(f_row[0])
            if not os.path.exists(path): continue
            
            try:
                # 2. Load Image
                img = cv2.imread(path)
                if img is None: continue
                
                # 3. Re-detect Faces
                faces = detector.face_app.get(img)
                
                # 4. FORCE SAVE (Bypassing Detector logic)
                for i, face in enumerate(faces):
                    bbox = face.bbox.astype(int)
                    # Crop: [y1:y2, x1:x2]
                    crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    
                    if crop.size > 0:
                        # Naming Convention: filename.jpg_0.jpg
                        # This matches what backend.py looks for
                        out_name = f"{os.path.basename(path)}_{i}.jpg"
                        out_path = os.path.join(FACES_DIR, out_name)
                        
                        cv2.imwrite(out_path, crop)
                        generated_count += 1
                
            except Exception as e:
                print(f"Error processing {path}: {e}")

    print(f"✅ Generated {generated_count} thumbnails.")
    print("👉 Refresh the UI at http://10.0.0.230:8001")

if __name__ == "__main__":
    main()