#!/usr/bin/env python3
import os
from huggingface_hub import snapshot_download
from insightface.app import FaceAnalysis
from ultralytics import YOLOWorld

# Define Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def download_repo(repo_id, folder_name):
    target_path = os.path.join(MODELS_DIR, folder_name)
    print(f"⬇️  Syncing {repo_id} to {target_path}...")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_path,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.h5", "*.msgpack", "*.bin"] 
        )
        print(f"✅ Verified {folder_name}")
    except Exception as e:
        print(f"❌ Error downloading {repo_id}: {e}")

def main():
    print("🚀 Starting Model Sync...\n")
    
    # 1. YOLO-World (The New Detector)
    # Ultralytics handles its own caching, but we can force a download to be sure.
    print("⬇️  Checking YOLO-World (v8 Large)...")
    try:
        # This downloads 'yolov8l-worldv2.pt' to the current directory or cache
        model = YOLOWorld('yolov8l-worldv2.pt')
        
        # Move it to our models folder for consistency (Optional, but clean)
        src_path = 'yolov8l-worldv2.pt'
        dst_path = os.path.join(MODELS_DIR, 'yolov8l-worldv2.pt')
        
        if os.path.exists(src_path):
            os.rename(src_path, dst_path)
            print(f"✅ Moved YOLO weights to {dst_path}")
        elif os.path.exists(dst_path):
            print(f"✅ YOLO weights already present in {dst_path}")
        else:
            print("⚠️  YOLO downloaded to system cache (default behavior).")
            
    except Exception as e:
        print(f"❌ Error downloading YOLO: {e}")

    # 2. Llama 3.2 Vision (The Captioner)
    download_repo("meta-llama/Llama-3.2-11B-Vision-Instruct", "llama_3_2_vision")

    # 3. InsightFace (Face Detection)
    try:
        print("⬇️  Checking InsightFace...")
        app = FaceAnalysis(name='buffalo_l')
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("✅ InsightFace Ready")
    except Exception as e:
        print(f"❌ InsightFace Error: {e}")

    print("\n🎉 All models synced!")

if __name__ == "__main__":
    main()