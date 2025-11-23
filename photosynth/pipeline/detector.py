import cv2
import os
import torch
import json
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from ultralytics import YOLOWorld

class Detector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.face_app = None
        self.yolo_model = None
        
        # Paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.vocab_path = os.path.join(self.base_dir, "photosynth", "vocabulary.json")
        
        self._load_models()

    def _load_models(self):
        # 1. InsightFace
        print(f"[{self.device}] 🔍 Loading InsightFace...")
        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        # 2. YOLO-World
        print(f"[{self.device}] 🦅 Loading YOLO-World...")
        # It auto-downloads 'yolov8l-worldv2.pt' to current dir or cache on first run
        self.yolo_model = YOLOWorld('yolov8l-worldv2.pt')
        self.yolo_model.to(self.device)
        
        # 3. Load Vocabulary
        if os.path.exists(self.vocab_path):
            with open(self.vocab_path, 'r') as f:
                vocab = json.load(f)
            print(f"   📚 Loaded {len(vocab)} classes from vocabulary.json")
        else:
            print("   ⚠️ vocabulary.json not found! Using fallback list.")
            vocab = ["person", "cat", "dog", "car", "food"]

        # Compile vocabulary into the model (Offline Optimization)
        self.yolo_model.set_classes(vocab)

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
        
        # Face
        faces = self.face_app.get(image_cv)
        self._save_face_crops(faces, image_cv, image_path)
        
        # YOLO (conf=0.05 is better for massive vocabulary lists)
        results = self.yolo_model.predict(image_path, conf=0.05, verbose=False)
        
        objs = set()
        for r in results:
            for c in r.boxes.cls:
                objs.add(self.yolo_model.names[int(c)])

        return {
            "status": "SUCCESS",
            "faces": [{"det_score": float(f.det_score)} for f in faces],
            "objects": list(objs),
            "is_video": False
        }

    def _process_video(self, video_path):
        print(f"🎬 Video detected. Sampling...")
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
        
        all_objects = set()
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_idx % frame_interval == 0:
                # YOLO handles numpy arrays (frames) natively
                results = self.yolo_model.predict(frame, conf=0.05, verbose=False)
                for r in results:
                    for c in r.boxes.cls:
                        all_objects.add(self.yolo_model.names[int(c)])
            
            frame_idx += 1
            
        cap.release()
        return {
            "status": "SUCCESS",
            "faces": [], 
            "objects": list(all_objects),
            "is_video": True
        }

    def _save_face_crops(self, faces, img, image_path):
        faces_dir = os.path.join(self.base_dir, "faces_crop")
        os.makedirs(faces_dir, exist_ok=True)
        for i, face in enumerate(faces):
            bbox = face.bbox.astype(int)
            crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            if crop.size > 0:
                cv2.imwrite(os.path.join(faces_dir, f"{os.path.basename(image_path)}_{i}.jpg"), crop)