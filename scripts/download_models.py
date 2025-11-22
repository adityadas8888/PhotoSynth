# ~/personal/PhotoSynth/scripts/download_models.py
import os
from huggingface_hub import snapshot_download

MODELS_DIR = os.path.expanduser("~/personal/PhotoSynth/models")

def download_model(repo_id, folder_name):
    print(f"⬇️  Downloading {repo_id} to {folder_name}...")
    try:
        snapshot_download(
            repo_id=repo_id, 
            local_dir=os.path.join(MODELS_DIR, folder_name),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx"]
        )
        print(f"✅ {folder_name} ready.\n")
    except Exception as e:
        print(f"❌ Error {repo_id}: {e}\n")

def main():
    # 1. Llama 3.2 Vision (11B) - For 3090 PC (Day-to-Day)
    download_model("meta-llama/Llama-3.2-11B-Vision-Instruct", "llama_3_2_vision")

    # 2. Qwen2-VL-7B-Instruct - For 5090 PC (Backlog/OCR Powerhouse)
    # Note: We use 7B because 72B is too big even for 5090 without complex splitting
    download_model("Qwen/Qwen2-VL-7B-Instruct", "qwen2_vl")

    # 3. Detection Models (For 3090)
    download_model("IDEA-Research/grounding-dino-base", "grounding_dino")
    download_model("facebook/sam2.1-hiera-large", "sam_3")
    download_model("PaddlePaddle/PaddleOCR-VL-0.9B", "paddleocr_vl")

if __name__ == "__main__":
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    main()