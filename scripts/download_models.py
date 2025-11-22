# ~/personal/PhotoSynth/scripts/download_models.py
import os
from huggingface_hub import snapshot_download

MODELS_DIR = os.path.expanduser("~/personal/PhotoSynth/models")

def download_model(repo_id, folder_name):
    target_dir = os.path.join(MODELS_DIR, folder_name)
    if os.path.exists(target_dir) and os.listdir(target_dir):
        print(f"✅ {folder_name} already exists. Skipping.")
        return

    print(f"⬇️  Downloading {repo_id} to {folder_name}...")
    try:
        snapshot_download(
            repo_id=repo_id, 
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx"]
        )
        print(f"✅ {folder_name} ready.\n")
    except Exception as e:
        print(f"❌ Error {repo_id}: {e}\n")

import argparse

def main():
    parser = argparse.ArgumentParser(description="Download PhotoSynth Models")
    parser.add_argument("--model", type=str, help="Specific model key to download (e.g., sam_3, qwen3)")
    args = parser.parse_args()

    print(f"📂 Saving models to: {MODELS_DIR}\n")

    models_to_download = {
        "llama_3_2_vision": ("meta-llama/Llama-3.2-11B-Vision-Instruct", "llama_3_2_vision"),
        "qwen3_vl_32b": ("Qwen/Qwen3-VL-32B-Instruct", "qwen3_vl_32b"),
        "sam_3": ("facebook/sam3", "sam_3"),
        "grounding_dino": ("IDEA-Research/grounding-dino-base", "grounding_dino"),
        "paddleocr_vl": ("PaddlePaddle/PaddleOCR-VL", "paddleocr_vl"),
    }

    if args.model:
        if args.model in models_to_download:
            repo_id, folder = models_to_download[args.model]
            download_model(repo_id, folder)
        else:
            print(f"❌ Unknown model: {args.model}")
    else:
        # Download all
        for key, (repo_id, folder) in models_to_download.items():
            download_model(repo_id, folder)

if __name__ == "__main__":
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    main()