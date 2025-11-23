#!/usr/bin/env python3
import os
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoProcessor, 
    MllamaForConditionalGeneration
)
from insightface.app import FaceAnalysis

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def download_model(model_id, folder_name, model_class=AutoModelForCausalLM, **kwargs):
    target_path = os.path.join(MODELS_DIR, folder_name)
    if os.path.exists(target_path) and os.listdir(target_path):
        print(f"✅ {folder_name} exists. Skipping.")
        return

    print(f"⬇️ Downloading {model_id}...")
    try:
        AutoProcessor.from_pretrained(model_id, trust_remote_code=True).save_pretrained(target_path)
        model_class.from_pretrained(model_id, trust_remote_code=True, **kwargs).save_pretrained(target_path)
        print(f"🎉 Saved to {target_path}")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    # 1. Florence-2 (Object Detection)
    download_model("microsoft/Florence-2-large", "florence_2_large", torch_dtype=torch.float16)

    # 2. Llama 3.2 Vision (Node A VLM)
    download_model("meta-llama/Llama-3.2-11B-Vision-Instruct", "llama_3_2_vision", model_class=MllamaForConditionalGeneration)

    # 3. InsightFace
    try:
        print("⬇️ Checking InsightFace...")
        app = FaceAnalysis(name='buffalo_l')
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("✅ InsightFace Ready")
    except Exception as e:
        print(f"❌ InsightFace Error: {e}")

if __name__ == "__main__":
    main()