import socket
import torch
import json
import os
import cv2  # <--- Added
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
        # ... (Keep your existing loading logic exactly as is) ...
        # ... (Paste your existing _load_model method here) ...
        # (Just ensuring I don't delete your setup)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        # Node B (5090)
        if "5090" in self.hostname:
            self.model_type = "Qwen3"
            model_path = os.path.join(MODELS_DIR, "qwen3_vl_32b")
            print(f"[{self.hostname}] 🚀 Loading Qwen3-VL from {model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, quantization_config=bnb_config, device_map="auto", trust_remote_code=True, local_files_only=True
            )
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        
        # Node A (3090)
        else:
            self.model_type = "Llama"
            model_path = os.path.join(MODELS_DIR, "llama_3_2_vision")
            print(f"[{self.hostname}] 🌿 Loading Llama 3.2 from {model_path}")
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_path, quantization_config=bnb_config, device_map="auto", local_files_only=True
            )
            self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

    def _load_image_or_video(self, file_path):
        """Helper to load an image or extract a frame from a video."""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in ['.mp4', '.mov', '.avi', '.mkv', '.m4v']:
            # Extract frame from video
            cap = cv2.VideoCapture(file_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Jump to middle
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Convert BGR to RGB
                return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                print(f"⚠️ Failed to extract frame from video: {file_path}")
                # Fallback: Create a blank black image to prevent crash
                return Image.new('RGB', (224, 224), color='black')
        else:
            # Standard Image
            return Image.open(file_path)

    def generate_analysis(self, image_path, det_results=None):
        if det_results is None: det_results = {}
        
        # Parse Context
        faces = det_results.get('faces', [])
        objects = det_results.get('objects', [])
        
        context_str = ""
        if faces: context_str += f"Contains {len(faces)} people. "
        if objects: context_str += f"Key objects: {', '.join(objects[:5])}. "

        prompt = (
            f"Context: {context_str}\n"
            "Task: Write a single, concise caption for this photo/video (max 200 characters).\n"
            "Rules: No bullet points. No intro. Focus on main subject/action.\n"
            "Example: 'A woman in a leopard-print bikini stands on the beach looking at the Golden Gate Bridge.'"
            "\n\nAlso provide 5-10 JSON keywords."
        )

        print(f"[{self.hostname}] 🧠 Generating caption...")
        
        if self.model_type == "Qwen3":
            return self._generate_qwen(image_path, prompt)
        else:
            return self._generate_llama(image_path, prompt)

    def _generate_llama(self, image_path, prompt_text):
        # FIX: Use the helper logic
        image = self._load_image_or_video(image_path)
        
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        
        inputs = self.processor(
            image, 
            input_text, 
            add_special_tokens=False, 
            return_tensors="pt"
        ).to(self.model.device)

        output = self.model.generate(**inputs, max_new_tokens=512)
        
        # Trim prompt
        input_len = inputs.input_ids.shape[1]
        generated_ids = output[:, input_len:]
        raw_text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        
        return self._parse_output(raw_text)

    # ... (Keep _generate_qwen and _parse_output as they were) ...
    def _parse_output(self, raw_text):
        # (Your existing parse logic)
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

    def _generate_qwen(self, image_path, prompt_text):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text[0]

    def _generate_llama(self, image_path, prompt_text):
        image = Image.open(image_path)
        
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text}
            ]}
        ]
        
        # Prepare inputs
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            image, 
            input_text, 
            add_special_tokens=False, 
            return_tensors="pt"
        ).to(self.model.device)

        # Generate
        output_ids = self.model.generate(**inputs, max_new_tokens=512)

        # FIX: Trim the input tokens so we only get the new generated text
        input_len = inputs.input_ids.shape[1]
        generated_ids = output_ids[:, input_len:]

        # Decode only the new tokens
        return self.processor.decode(generated_ids[0], skip_special_tokens=True)