import cv2
import os
import torch
import sys
from unittest.mock import MagicMock
from PIL import Image
from insightface.app import FaceAnalysis

# --- 🛡️ CRITICAL FIX: MOCK FLASH ATTENTION ---
# We inject this BEFORE importing transformers so the remote code loads safely.
# This forces Florence-2 to use PyTorch's internal SDPA (which is supported).
if "flash_attn" not in sys.modules:
    sys.modules["flash_attn"] = MagicMock()

# NOW import transformers
from transformers import AutoProcessor, AutoModelForCausalLM 

class Detector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.face_app = None
        self.florence_model = None
        self.florence_processor = None
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.models_dir = os.path.join(self.base_dir, "models")
        self._load_models()

    def _load_models(self):
        # 1. InsightFace
        print(f"[{self.device}] 🔍 Loading InsightFace...")
        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        # 2. Florence-2-Large
        print(f"[{self.device}] 💃 Loading Florence-2-Large...")
        model_path = os.path.join(self.models_dir, "florence_2_large")
        
        # Load Local Only
        self.florence_model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            local_files_only=True, 
            torch_dtype=torch.float16
        ).to(self.device)
        
        self.florence_processor = AutoProcessor.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            local_files_only=True
        )

    def run_detection(self, file_path):
        print(f"Processing {os.path.basename(file_path)}...")
        
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.mp4', '.mov', '.avi', '.mkv', '.m4v']:
            return self._process_video(file_path)
        else:
            return self._process_image(file_path)

    def _process_image(self, image_path):
        image_cv = cv2.imread(image_path)
        if image_cv is None: return {}
        image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        
        faces = self.face_app.get(image_cv)
        objs = self._run_florence_on_frame(image_pil)
        
        self._save_face_crops(faces, image_cv, image_path)

        return {
            "status": "SUCCESS",
            "faces": [{"det_score": float(f.det_score)} for f in faces],
            "objects": objs,
            "is_video": False
        }

    def _process_video(self, video_path):
        print(f"🎬 Video detected. Calculating sampling interval...")
        cap = cv2.VideoCapture(video_path)
        
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = raw_fps if raw_fps > 0 else 30.0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        # 5-2-1 Logic
        if duration > 30: interval_sec = 5
        elif duration > 5: interval_sec = 2
        else: interval_sec = 1
            
        frame_interval = int(fps * interval_sec)
        if frame_interval < 1: frame_interval = 1
        
        print(f"   Duration: {duration:.1f}s -> Sampling every {interval_sec}s")
        
        all_objects = set()
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_idx % frame_interval == 0:
                image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                objs = self._run_florence_on_frame(image_pil)
                all_objects.update(objs)
            
            frame_idx += 1
            
        cap.release()
        return {
            "status": "SUCCESS",
            "faces": [], 
            "objects": list(all_objects),
            "is_video": True
        }

    def _run_florence_on_frame(self, image_pil):
        task_prompt = "<OD>"
        inputs = self.florence_processor(text=task_prompt, images=image_pil, return_tensors="pt").to(self.device, torch.float16)
        generated_ids = self.florence_model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
            max_new_tokens=1024, do_sample=False, num_beams=3
        )
        generated_text = self.florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.florence_processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(image_pil.width, image_pil.height)
        )
        return [label.lower() for label in parsed.get('<OD>', {}).get('labels', [])]

    def _save_face_crops(self, faces, img, image_path):
        faces_dir = os.path.join(self.base_dir, "faces_crop")
        os.makedirs(faces_dir, exist_ok=True)
        for i, face in enumerate(faces):
            bbox = face.bbox.astype(int)
            crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            if crop.size > 0:
                cv2.imwrite(os.path.join(faces_dir, f"{os.path.basename(image_path)}_{i}.jpg"), crop)