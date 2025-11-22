import socket
import torch
import json
from transformers import AutoProcessor, AutoModelForCausalLM, MllamaForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image

class Captioner:
    def __init__(self):
        self.hostname = socket.gethostname()
        self.model = None
        self.processor = None
        self.model_type = None
        self._load_model()

    def _load_model(self):
        """
        Loads the appropriate model based on the hostname.
        Node B (5090) -> Qwen3-VL-32B (4-bit)
        Node A (3090) -> Llama 3.2 Vision (4-bit)
        """
        if "5090" in self.hostname:
            self.model_type = "Qwen3"
            print(f"[{self.hostname}] 🚀 Loading Qwen3-VL-32B (4-bit)...")
            model_id = "Qwen/Qwen3-VL-32B-Instruct"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype="auto",
                device_map="auto",
                quantization_config={"load_in_4bit": True}
            )
            self.processor = AutoProcessor.from_pretrained(model_id)

        else:
            self.model_type = "Llama"
            print(f"[{self.hostname}] 🌿 Loading Llama 3.2 Vision (11B, 4-bit)...")
            model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
            
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                load_in_4bit=True, 
            )
            self.processor = AutoProcessor.from_pretrained(model_id)

    def generate_analysis(self, image_path):
        """
        Generates both a detailed narrative and search-optimized concepts.
        Returns a dict: {'narrative': str, 'concepts': list}
        """
        print(f"[{self.hostname}] Analyzing {image_path} using {self.model_type}...")
        
        # We ask for a structured output to get both narrative and keywords
        prompt = (
            "Analyze this image in detail.\n"
            "1. Provide a rich, detailed narrative description of the scene, lighting, and subjects.\n"
            "2. Provide a list of 5-10 critical search concepts/keywords. "
            "Use synonym expansion (e.g., if you see a 'car', add 'vehicle', 'sedan'). "
            "Format the keywords as a JSON list."
        )

        if self.model_type == "Qwen3":
            raw_output = self._generate_qwen(image_path, prompt)
        else:
            raw_output = self._generate_llama(image_path, prompt)
            
        return self._parse_output(raw_output)

    def _parse_output(self, raw_text):
        """
        Parses the VLM output to extract narrative and concepts.
        This is a heuristic parser since VLMs might not output perfect JSON.
        """
        narrative = raw_text
        concepts = []
        
        # Simple heuristic: Look for the JSON list part
        try:
            # Find start and end of list
            start = raw_text.find('[')
            end = raw_text.rfind(']') + 1
            if start != -1 and end != -1:
                json_str = raw_text[start:end]
                concepts = json.loads(json_str)
                # Remove the JSON part from the narrative to keep it clean
                narrative = raw_text.replace(json_str, "").strip()
        except Exception as e:
            print(f"⚠️ Failed to parse keywords from VLM output: {e}")
            # Fallback: Use the whole text as narrative, empty concepts
            
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
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
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
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    def _generate_llama(self, image_path, prompt_text):
        image = Image.open(image_path)
        
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text}
            ]}
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)

        output = self.model.generate(**inputs, max_new_tokens=512)
        return self.processor.decode(output[0])
