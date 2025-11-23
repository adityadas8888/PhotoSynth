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
    BitsAndBytesConfig
)
from PIL import Image

# Load Secrets & Paths
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
        # 4-bit Config for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        # Node B (5090) -> Qwen
        if "5090" in self.hostname:
            self.model_type = "Qwen3"
            model_path = os.path.join(MODELS_DIR, "qwen3_vl_32b")
            print(f"[{self.hostname}] 🚀 Loading Qwen3-VL from {model_path}")
            # Fallback to hub if local missing
            load_path = model_path if os.path.exists(model_path) else "Qwen/Qwen2.5-VL-7B-Instruct"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                load_path, 
                quantization_config=bnb_config, 
                device_map="auto", 
                trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(load_path, trust_remote_code=True)
        
        # Node A (3090) -> Llama
        else:
            self.model_type = "Llama"
            model_path = os.path.join(MODELS_DIR, "llama_3_2_vision")
            print(f"[{self.hostname}] 🌿 Loading Llama 3.2 from {model_path}")
            # Fallback to hub if local missing
            load_path = model_path if os.path.exists(model_path) else "meta-llama/Llama-3.2-11B-Vision-Instruct"
            
            self.model = MllamaForConditionalGeneration.from_pretrained(
                load_path, 
                quantization_config=bnb_config, 
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(load_path)

    def _load_image_or_video(self, file_path):
        """Extracts a frame from video or loads image."""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in ['.mp4', '.mov', '.avi', '.mkv', '.m4v']:
            # Video Strategy: Grab the middle frame
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
                print(f"⚠️ Could not read frame from {file_path}")
                return Image.new('RGB', (224, 224), 'black')
        else:
            return Image.open(file_path)

    def generate_analysis(self, image_path, det_results=None):
        if det_results is None: det_results = {}
        
        # Parse Context
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
            return {"narrative": "Error generating caption.", "concepts": []}

    def _parse_output(self, raw_text):
        """Extracts JSON concepts and clean narrative from the raw model output."""
        narrative = raw_text
        concepts = []
        try:
            # Find JSON list [ ... ]
            start = raw_text.find('[')
            if start != -1:
                # Use raw_decode to grab just the JSON part
                decoder = json.JSONDecoder()
                concepts, end_pos = decoder.raw_decode(raw_text[start:])
                # Remove the JSON string from the narrative
                narrative = raw_text[:start] + raw_text[start + end_pos:]
        except Exception as e:
            print(f"⚠️ JSON Parse Warning: {e}")
            
        return {"narrative": narrative.strip(), "concepts": concepts}

    def _generate_llama(self, image_path, prompt_text):
        image = self._load_image_or_video(image_path)
        
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        
        inputs = self.processor(
            image, 
            input_text, 
            add_special_tokens=False, 
            return_tensors="pt"
        ).to(self.model.device)

        output = self.model.generate(**inputs, max_new_tokens=256)
        
        # Trim the prompt so we don't repeat it
        input_len = inputs.input_ids.shape[1]
        generated_ids = output[:, input_len:]
        raw_text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        
        # CRITICAL FIX: Always return the dictionary, not the string
        return self._parse_output(raw_text)

    def _generate_qwen(self, image_path, prompt_text):
        # Qwen implementation (assuming simple generation for now)
        # Since Qwen handles video paths natively, we can pass path or frames.
        # For consistency, let's use the frame logic unless Qwen requires paths.
        # Placeholder if using Qwen-VL-Chat logic:
        image = self._load_image_or_video(image_path)
        
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image}, # Pass PIL image
                {"type": "text", "text": prompt_text}
            ]}
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        raw_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        return self._parse_output(raw_text)