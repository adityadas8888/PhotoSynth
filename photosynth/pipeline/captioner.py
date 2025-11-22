import socket
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, MllamaForConditionalGeneration
from qwen_vl_utils import process_vision_info

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
            # VRAM Calculation for 32B Model @ 4-bit:
            # Weights: 32B * 0.5 bytes = 16GB
            # KV Cache + Activations (approx): ~4-6GB
            # Total Est: ~20-22GB.
            # RTX 5090 has 32GB VRAM -> Fits comfortably with ~10GB headroom.
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

    def generate_caption(self, image_path):
        """
        Generates a caption for the given image path.
        """
        print(f"[{self.hostname}] Generating caption for {image_path} using {self.model_type}...")
        
        if self.model_type == "Qwen3":
            return self._generate_qwen(image_path)
        else:
            return self._generate_llama(image_path)

    def _generate_qwen(self, image_path):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Describe this image in detail."},
                ],
            }
        ]
        
        # Preparation for inference
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

        # Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    def _generate_llama(self, image_path):
        from PIL import Image
        image = Image.open(image_path)
        
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image in detail."}
            ]}
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)

        output = self.model.generate(**inputs, max_new_tokens=128)
        return self.processor.decode(output[0])
