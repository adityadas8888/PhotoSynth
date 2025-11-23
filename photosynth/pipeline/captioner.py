import socket
import torch
import json
import os
import cv2
from dotenv import load_dotenv
from transformers import (
    AutoProcessor, 
    AutoModelForCausalLM, 
    MllamaForConditionalGeneration,
    # Qwen2VLForConditionalGeneration, # OLD
    Qwen3VLForConditionalGeneration,   # <--- NEW: Import the Qwen3 Class
    BitsAndBytesConfig
)
from qwen_vl_utils import process_vision_info 
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SECRETS_PATH = os.path.join(BASE_DIR, ".secretsenv")
load_dotenv(SECRETS_PATH)
MODELS_DIR = os.path.join(BASE_DIR, "models")

class Captioner:
    def __init__(self):
        self.hostname = socket.gethostname()
        self.model = None
        self.processor = None
        self.model_type = None
        self._load_model()

    def _load_model(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        # --- Node B (5090) -> Qwen3-VL ---
        if "5090" in self.hostname:
            self.model_type = "Qwen3"
            model_path = os.path.join(MODELS_DIR, "qwen3_vl_32b")
            
            # Fallback to Hub ID if local is missing
            # Note: Using the 7B Instruct version as a safe default if 32B isn't there
            load_path = model_path if os.path.exists(model_path) else "Qwen/Qwen3-VL-7B-Instruct"
            
            print(f"[{self.hostname}] 🚀 Loading Qwen3-VL from {load_path}")
            
            # FIX: Use Qwen3VLForConditionalGeneration
            # AutoModelForCausalLM fails because it doesn't know how to handle the vision encoder config
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                load_path, 
                quantization_config=bnb_config, 
                device_map="auto", 
                trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(load_path, trust_remote_code=True)
        
        # --- Node A (3090) -> Llama ---
        else:
            self.model_type = "Llama"
            model_path = os.path.join(MODELS_DIR, "llama_3_2_vision")
            print(f"[{self.hostname}] 🌿 Loading Llama 3.2 from {model_path}")
            
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_path, 
                quantization_config=bnb_config, 
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(model_path)

    def _load_image_or_video(self, file_path):
        """Extracts a frame from video or loads image (With Path Auto-Correction)."""
        
        # --- 1. AUTO-CORRECT PATH ---
        if not os.path.exists(file_path):
            # Path from 3090 likely looks like: /home/aditya/personal/nas/...
            # Path on 5090 is: /home/adityadas/personal/nas/...
            
            # Strategy: Find the part after "nas" and append it to OUR nas path.
            if "personal/nas" in file_path:
                relative_part = file_path.split("personal/nas")[-1]
                # Construct local path based on CURRENT user
                current_home = os.path.expanduser("~")
                corrected_path = os.path.join(current_home, "personal/nas", relative_part.strip("/"))
                
                if os.path.exists(corrected_path):
                    print(f"🔄 Path Remapped: {file_path} -> {corrected_path}")
                    file_path = corrected_path
                else:
                    print(f"❌ Path correction failed. Local path does not exist: {corrected_path}")
                    return Image.new('RGB', (224, 224), 'black') # Fail gracefully

        # --- 2. Load Content (Video or Image) ---
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in ['.mp4', '.mov', '.avi', '.mkv', '.m4v']:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened(): 
                print(f"⚠️ Error opening video: {file_path}")
                return Image.new('RGB', (224, 224), 'black')

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return Image.new('RGB', (224, 224), 'black')
        else:
            return Image.open(file_path)

    def generate_analysis(self, image_path, det_results=None):
        if det_results is None: det_results = {}
        faces = det_results.get('faces', [])
        objects = det_results.get('objects', [])
        
        context_str = ""
        if faces: context_str += f"Contains {len(faces)} people. "
        if objects: context_str += f"Key objects: {', '.join(objects[:7])}. "

        prompt = (
            f"Context: {context_str}\n"
            "Task: Write a concise caption (max 200 chars). No lists. No filler.\n"
            "Example: 'A cat sleeping on a sofa near a window.'\n"
            "Also provide 5-10 JSON keywords."
        )

        print(f"[{self.hostname}] 🧠 Generating caption for {os.path.basename(image_path)}...")
        try:
            if self.model_type == "Qwen3":
                return self._generate_qwen(image_path, prompt)
            else:
                return self._generate_llama(image_path, prompt)
        except Exception as e:
            print(f"❌ Caption Generation Error: {e}")
            return {"narrative": "Error.", "concepts": []}

    def _parse_output(self, raw_text):
        narrative = raw_text
        concepts = []
        try:
            start = raw_text.find('[')
            if start != -1:
                decoder = json.JSONDecoder()
                concepts, end_pos = decoder.raw_decode(raw_text[start:])
                narrative = raw_text[:start] + raw_text[start + end_pos:]
        except Exception: pass
        return {"narrative": narrative.strip(), "concepts": concepts}

    def _generate_llama(self, image_path, prompt_text):
        image = self._load_image_or_video(image_path)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(image, input_text, add_special_tokens=False, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=256)
        input_len = inputs.input_ids.shape[1]
        return self._parse_output(self.processor.decode(output[:, input_len:][0], skip_special_tokens=True))

    def _generate_qwen(self, image_path, prompt_text):
        image = self._load_image_or_video(image_path)
        
        # Qwen3-VL structure
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text}
            ]}
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text], 
            images=image_inputs, 
            videos=video_inputs, 
            padding=True, 
            return_tensors="pt"
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        
        # Qwen3 trimming logic
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        raw_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return self._parse_output(raw_text)