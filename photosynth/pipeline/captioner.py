import socket
import torch
import json
import os
from dotenv import load_dotenv  # <--- Added
from transformers import (
    AutoProcessor, 
    AutoModelForCausalLM, 
    MllamaForConditionalGeneration,
    BitsAndBytesConfig          # <--- Added
)
from qwen_vl_utils import process_vision_info
from PIL import Image

# Load the secrets file explicitly
# We look for .secretsenv in the project root (2 levels up from this file)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SECRETS_PATH = os.path.join(BASE_DIR, ".secretsenv")
load_dotenv(SECRETS_PATH)

class Captioner:
    def __init__(self):
        self.hostname = socket.gethostname()
        self.model = None
        self.processor = None
        self.model_type = None
        self._load_model()

    def _load_model(self):
        # Check for Token
        if not os.getenv("HF_TOKEN"):
            print("⚠️ WARNING: HF_TOKEN not found in .secretsenv. Llama models may hang!")

        # Define the new 4-bit config (Fixes the deprecation warning)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        if "5090" in self.hostname:
            self.model_type = "Qwen3"
            print(f"[{self.hostname}] 🚀 Loading Qwen3-VL-32B (4-bit)...")
            model_id = "Qwen/Qwen3-VL-32B-Instruct"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True 
            )
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        else:
            # Node A (3090)
            self.model_type = "Llama"
            print(f"[{self.hostname}] 🌿 Loading Llama 3.2 Vision (11B, 4-bit)...")
            model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
            
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                token=os.getenv("HF_TOKEN")  # <--- Explicitly pass token
            )
            self.processor = AutoProcessor.from_pretrained(
                model_id, 
                token=os.getenv("HF_TOKEN")
            )