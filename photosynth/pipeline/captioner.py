import socket
import torch
import json
import os
import cv2
import re
from dotenv import load_dotenv
from transformers import (
    AutoProcessor, 
    AutoModelForCausalLM, 
    MllamaForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    BitsAndBytesConfig
)
from qwen_vl_utils import process_vision_info 
from PIL import Image
from photosynth.utils.paths import heal_path

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

        if "5090" in self.hostname:
            self.model_type = "Qwen3"
            model_path = os.path.join(MODELS_DIR, "qwen3_vl_32b")
            load_path = model_path if os.path.exists(model_path) else "Qwen/Qwen3-VL-7B-Instruct"
            print(f"[{self.hostname}] 🚀 Loading Qwen3-VL from {load_path}")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                load_path, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(load_path, trust_remote_code=True)
        else:
            self.model_type = "Llama"
            model_path = os.path.join(MODELS_DIR, "llama_3_2_vision")
            print(f"[{self.hostname}] 🌿 Loading Llama 3.2 from {model_path}")
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_path, quantization_config=bnb_config, device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(model_path)

    def _load_image_or_video(self, file_path):
        """Extracts a frame from video or loads image (With Path Auto-Correction)."""
        file_path = heal_path(file_path)

        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.mp4', '.mov', '.avi', '.mkv', '.m4v']:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened(): return Image.new('RGB', (224, 224), 'black')
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
            ret, frame = cap.read()
            cap.release()
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if ret else Image.new('RGB', (224, 224), 'black')
        else:
            return Image.open(file_path)

    def generate_analysis(self, image_path, det_results=None):
        if det_results is None: det_results = {}
        
        # 1. Build Context
        faces = det_results.get('faces', [])
        objects = det_results.get('objects', [])
        known_people = det_results.get('known_people', []) # e.g. ['Aditya']
        
        context_parts = []
        
        # FORCE NAME USAGE
        if known_people:
            names = ", ".join(known_people)
            context_parts.append(f"This image contains specific people you know: {names}. You MUST refer to them by name in the caption.")
        elif faces:
            context_parts.append(f"Contains {len(faces)} unidentified people.")
            
        if objects:
            context_parts.append(f"Key objects present: {', '.join(objects[:7])}.")

        context_str = " ".join(context_parts)

        # 2. Strict Prompting
        prompt = (
            f"Context: {context_str}\n"
            "Task: Analyze the image and provide a structured JSON response.\n"
            "Constraints:\n"
            "1. 'caption': A single, concise sentence (MAX 200 CHARACTERS). Be descriptive but brief.\n"
            "2. 'keywords': A list of relevant tags/keywords.\n"
            "3. If names are provided in Context, **USE THEM** in the caption.\n"
            "4. Output MUST be valid JSON only. No markdown, no explanations.\n\n"
            "JSON Schema:\n"
            '{"caption": "string", "keywords": ["string", "string"]}'
        )

        print(f"[{self.hostname}] 🧠 Generating caption for {os.path.basename(image_path)}...")
        
        try:
            if self.model_type == "Qwen3":
                raw = self._generate_qwen(image_path, prompt)
            else:
                raw = self._generate_llama(image_path, prompt)
            
            result = self._parse_output(raw)
            
            print(f"   📝 Caption: {result['narrative']}")
            print(f"   🏷️  Tags:    {result['concepts']}")
            return result

        except Exception as e:
            print(f"❌ Caption Generation Error: {e}")
            return {"narrative": "Error.", "concepts": []}

    def _parse_output(self, raw_text):
        """Aggressive cleaner for LLM output."""
        narrative = ""
        concepts = []
        
        try:
            # 1. Clean Markdown Code Blocks
            clean_text = raw_text.replace("```json", "").replace("```", "").strip()
            
            # 2. Find JSON Object
            start = clean_text.find('{')
            end = clean_text.rfind('}') + 1
            
            if start != -1 and end != -1:
                json_str = clean_text[start:end]
                data = json.loads(json_str)
                narrative = data.get("caption", "")
                concepts = data.get("keywords", [])
            else:
                # Fallback: Try to parse manually if model refused JSON
                narrative = clean_text
        except Exception:
            # Extreme Fallback
            narrative = raw_text[:200]

        # 3. Final Polish (Remove lang tags if they leaked)
        narrative = re.sub(r'lang="[^"]+"', '', narrative).strip()
        
        return {"narrative": narrative, "concepts": concepts}

    def _generate_llama(self, image_path, prompt_text):
        image = self._load_image_or_video(image_path)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(image, input_text, add_special_tokens=False, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=256)
        input_len = inputs.input_ids.shape[1]
        return self.processor.decode(output[:, input_len:][0], skip_special_tokens=True)

    def _generate_qwen(self, image_path, prompt_text):
        image = self._load_image_or_video(image_path)
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]