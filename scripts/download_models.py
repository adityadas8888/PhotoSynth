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

def main():
    print(f"📂 Saving models to: {MODELS_DIR}\n")

    # --- FOR 3090 PC (Daily Driver) ---
    # Llama 3.2 11B: Best for natural descriptions of people/scenes
    download_model("meta-llama/Llama-3.2-11B-Vision-Instruct", "llama_3_2_vision")

    # --- FOR 5090 PC (Backlog Beast) ---
    # Qwen3-VL-32B: The SOTA for OCR and details. 
    # Note: This is huge. We will run it in 4-bit on the 5090.
    download_model("Qwen/Qwen3-VL-32B-Instruct", "qwen3_vl_32b")

    # --- SHARED DETECTION MODELS (Run on 3090 usually) ---
    download_model("facebook/sam3", "sam_3")
    download_model("IDEA-Research/grounding-dino-base", "grounding_dino")
    download_model("PaddlePaddle/PaddleOCR-VL", "paddleocr_vl")

if __name__ == "__main__":
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    main()