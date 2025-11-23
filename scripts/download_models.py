#!/usr/bin/env python3
import os
from huggingface_hub import snapshot_download
from insightface.app import FaceAnalysis

# Define Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def download_repo(repo_id, folder_name):
    target_path = os.path.join(MODELS_DIR, folder_name)
    
    print(f"⬇️  Syncing {repo_id} to {target_path}...")
    try:
        # snapshot_download handles resumes and updates automatically
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_path,
            local_dir_use_symlinks=False,  # Actual files, not links
            ignore_patterns=["*.h5", "*.msgpack", "*.bin"] # Skip non-safetensors if possible to save space
        )
        print(f"✅ Verified {folder_name}")
    except Exception as e:
        print(f"❌ Error downloading {repo_id}: {e}")

def main():
    print("🚀 Starting Artifact Sync...\n")
    
    # 1. Florence-2 (Community Version)
    # Using snapshot_download bypasses the "Unrecognized Config" / Flash Attn errors
    download_repo("florence-community/Florence-2-large", "florence_2_large")

    # 2. Llama 3.2 Vision
    # Requires HF_TOKEN login
    download_repo("meta-llama/Llama-3.2-11B-Vision-Instruct", "llama_3_2_vision")

    # 3. InsightFace (Buffalo_L)
    try:
        print("⬇️  Checking InsightFace...")
        # InsightFace manages its own cache, usually in ~/.insightface
        app = FaceAnalysis(name='buffalo_l')
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("✅ InsightFace Ready")
    except Exception as e:
        print(f"❌ InsightFace Error: {e}")

    print("\n🎉 All models synced!")

if __name__ == "__main__":
    main()