#!/usr/bin/env python3
import os
import torch
from transformers import (
    AutoModelForCausalLM, # <--- CRITICAL: Must use this for Florence
    AutoProcessor, 
    MllamaForConditionalGeneration
)
from insightface.app import FaceAnalysis

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def download_model(model_id, folder_name, model_class, **kwargs):
    target_path = os.path.join(MODELS_DIR, folder_name)
    
    # Optional: Delete existing if you want to force re-download of the correct version
    # import shutil
    # if os.path.exists(target_path): shutil.rmtree(target_path)

    if os.path.exists(target_path) and os.listdir(target_path):
        print(f"✅ {folder_name} exists. Skipping.")
        return

    print(f"⬇️ Downloading {model_id}...")
    try:
        # Download Processor
        AutoProcessor.from_pretrained(model_id, trust_remote_code=True).save_pretrained(target_path)
        
        # Download Model
        model_class.from_pretrained(
            model_id, 
            trust_remote_code=True, 
            **kwargs
        ).save_pretrained(target_path)
        
        print(f"🎉 Saved to {target_path}")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Download required models for the pipeline."""
    
    # 1. Florence-2 (The Fix)
    # MUST use AutoModelForCausalLM, otherwise it loses the ability to generate text
    download_model(
        "florence-community/Florence-2-large",
        "florence_2_large",
        model_class=AutoModelForCausalLM, 
        torch_dtype=torch.float16  # Note: argument is 'torch_dtype', not 'dtype'
    )

    # 2. Llama 3.2 Vision
    download_model(
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "llama_3_2_vision",
        model_class=MllamaForConditionalGeneration,
        torch_dtype=torch.float16
    )

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