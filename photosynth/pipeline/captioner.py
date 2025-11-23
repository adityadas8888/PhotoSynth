import socket
import torch
import json
import os
from dotenv import load_dotenv
from transformers import (
    AutoProcessor, 
    AutoModelForCausalLM, 
    MllamaForConditionalGeneration,
    BitsAndBytesConfig
)
from qwen_vl_utils import process_vision_info
from PIL import Image

# 1. Load Secrets (Good practice to keep, even for local)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SECRETS_PATH = os.path.join(BASE_DIR, ".secretsenv")
load_dotenv(SECRETS_PATH)

# 2. Define Local Models Directory
# Resolves to ~/personal/PhotoSynth/models
MODELS_DIR = os.path.join(BASE_DIR, "models")

class Captioner:
    def __init__(self):
        self.hostname = socket.gethostname()
        self.model = None
        self.processor = None
        self.model_type = None
        self._load_model()

    def _load_model(self):
        # 4-bit Config (Optimized for 3090/5090)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        # --- NODE B (5090) ---
        if "5090" in self.hostname:
            self.model_type = "Qwen3"
            # Matches the folder you likely have on the 5090
            model_path = os.path.join(MODELS_DIR, "qwen3_vl_32b")
            
            print(f"[{self.hostname}] 🚀 Loading Qwen3-VL from local: {model_path}")
            
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=True
                )
                self.processor = AutoProcessor.from_pretrained(
                    model_path, 
                    trust_remote_code=True, 
                    local_files_only=True
                )
            except OSError:
                print(f"❌ Error: Model not found at {model_path}. Check folder name?")
                raise

        # --- NODE A (3090) ---
        else:
            self.model_type = "Llama"
            # Matches your ls output: 'llama_3_2_vision'
            model_path = os.path.join(MODELS_DIR, "llama_3_2_vision")
            
            print(f"[{self.hostname}] 🌿 Loading Llama 3.2 from local: {model_path}")
            
            try:
                self.model = MllamaForConditionalGeneration.from_pretrained(
                    model_path,
                    quantization_config=bnb_config,
                    device_map="auto",
                    local_files_only=True 
                )
                self.processor = AutoProcessor.from_pretrained(
                    model_path, 
                    local_files_only=True
                )
            except OSError as e:
                print(f"❌ Error: Model not found at {model_path}.")
                print(f"   Ensure the folder contains config.json and .safetensors files.")
                raise e

    def generate_analysis(self, image_path, det_results=None):
        if det_results is None: det_results = {}
        
        # Parse Context from Florence/InsightFace
        faces = det_results.get('faces', [])
        objects = det_results.get('objects', [])
        
        context_str = ""
        if faces:
            context_str += f"Contains {len(faces)} people. "
        if objects:
            # Only take top 5 objects to keep prompt clean
            context_str += f"Key objects: {', '.join(objects[:5])}. "

        # --- THE NEW PROMPT (Optimized for Synology Limits) ---
        prompt = (
            f"Context: {context_str}\n"
            "Task: Write a single, concise caption for this photo (max 200 characters).\n"
            "Rules:\n"
            "1. NO bullet points, NO lists, NO 'In summary'.\n"
            "2. Focus ONLY on the main subject and action.\n"
            "3. Do not start with 'The image shows' or 'lang=x-default'.\n"
            "4. Combine the context tags naturally.\n\n"
            "Example Output: 'A woman in a leopard-print bikini stands on the beach looking at the Golden Gate Bridge.'"
            "\n\nAlso provide 5-10 JSON keywords."
        )
        # -----------------------------------------------------

        print(f"[{self.hostname}] 🧠 Generating concise caption...")
        
        if self.model_type == "Qwen3":
            raw_output = self._generate_qwen(image_path, prompt)
        else:
            raw_output = self._generate_llama(image_path, prompt)
            
        return self._parse_output(raw_output)

    def _parse_output(self, raw_text):
        narrative = raw_text
        concepts = []
        try:
            start = raw_text.find('[')
            end = raw_text.rfind(']') + 1
            if start != -1 and end != -1:
                json_str = raw_text[start:end]
                concepts = json.loads(json_str)
                narrative = raw_text.replace(json_str, "").strip()
        except Exception as e:
            print(f"⚠️ Failed to parse keywords: {e}")
            
        return {"narrative": narrative, "concepts": concepts}

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