import cv2
import os
import torch
import json
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from ultralytics import YOLOWorld
from photosynth.utils.paths import heal_path

class Detector:
    def __init__(self, enable_yolo=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enable_yolo = enable_yolo
        self.face_app = None
        self.yolo_model = None
        
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.models_dir = os.path.join(self.base_dir, "models")
        self.vocab_path = os.path.join(self.base_dir, "photosynth", "vocabulary.json")
        
        self._load_models()

    def _load_models(self):
        print(f"[{self.device}] 🔍 Loading InsightFace...")
        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        if self.enable_yolo:
            print(f"[{self.device}] 🦅 Loading YOLO-World...")
            local_yolo = os.path.join(self.models_dir, 'yolov8l-worldv2.pt')
            self.yolo_model = YOLOWorld(local_yolo if os.path.exists(local_yolo) else 'yolov8l-worldv2.pt')
            self.yolo_model.to(self.device)
            
            if os.path.exists(self.vocab_path):
                with open(self.vocab_path, 'r') as f:
                    vocab = json.load(f)
                self.yolo_model.set_classes(vocab)



    def _identify_faces(self, faces):
        """Returns list of names ['Aditya', 'Ankita'] found in the image."""
        from photosynth.db import PhotoSynthDB
        try:
            db = PhotoSynthDB()
            known_faces = db.get_known_faces()
        except: return []

        if not known_faces: return []

        found_names = set()
        for face in faces:
            curr_emb = face.embedding
            best_score = 0.0
            best_name = None
            
            for _, name, known_emb in known_faces:
                score = np.dot(curr_emb, known_emb) / (np.linalg.norm(curr_emb) * np.linalg.norm(known_emb))
                if score > 0.55 and score > best_score: 
                    best_score = score
                    best_name = name
            
            if best_name and best_name != "Unknown":
                found_names.add(best_name)
        return list(found_names)

    def run_detection(self, file_path):
        print(f"Processing {os.path.basename(file_path)}...")
        ext = os.path.splitext(file_path)[1].lower()
        
        # VIDEO extensions
        if ext in ['.mp4', '.mov', '.avi', '.mkv', '.m4v']:
            return self._process_video(file_path)
        # IMAGE extensions (Added .webp)
        elif ext in ['.jpg', '.jpeg', '.png', '.arw', '.webp', '.heic']:
            return self._process_image(file_path)
        else:
            return {"status": "SKIPPED", "reason": "Unsupported format"}

    def _process_image(self, image_path):
        image_path = heal_path(image_path)
        image_cv = cv2.imread(image_path)
        if image_cv is None: return {}
        
        # 1. Faces
        faces = self.face_app.get(image_cv)
        known_people = self._identify_faces(faces)
        # self._save_face_crops(faces, image_cv, image_path)
        
        # 2. Objects
        objs = []
        if self.enable_yolo:
            results = self.yolo_model.predict(image_path, conf=0.05, verbose=False)
            for r in results:
                for c in r.boxes.cls:
                    objs.append(self.yolo_model.names[int(c)])
        
        return {
            "status": "SUCCESS",
            "faces": [f.embedding.tolist() for f in faces], # Embeddings present
            "face_count": len(faces),
            "known_people": known_people,
            "objects": list(set(objs)),
            "is_video": False
        }

    def _process_video(self, video_path):
        print(f"🎬 Video detected. Sampling...")
        video_path = heal_path(video_path)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened(): return {"status": "ERROR"}

        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = raw_fps if raw_fps > 0 else 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        if duration > 30: interval_sec = 5
        elif duration > 5: interval_sec = 2
        else: interval_sec = 1
            
        frame_interval = int(fps * interval_sec)
        if frame_interval < 1: frame_interval = 1
        
        all_objects = set()
        all_people = set()
        max_faces_seen_in_frame = 0
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_idx % frame_interval == 0:
                # Faces
                faces = self.face_app.get(frame)
                max_faces_seen_in_frame = max(max_faces_seen_in_frame, len(faces))
                names = self._identify_faces(faces)
                all_people.update(names)
                
                # Objects
                if self.enable_yolo:
                    results = self.yolo_model.predict(frame, conf=0.05, verbose=False)
                    for r in results:
                        for c in r.boxes.cls:
                            all_objects.add(self.yolo_model.names[int(c)])
            frame_idx += 1
            
        cap.release()
        
        return {
            "status": "SUCCESS",
            "faces": [], # Empty for video to avoid DB bloat
            "face_count": max_faces_seen_in_frame, # But we still report count!
            "known_people": list(all_people),
            "objects": list(all_objects),
            "is_video": True
        }

    def _save_face_crops(self, faces, img, image_path):
        # Only save if strictly necessary (usually only daily mode)
        if not self.enable_yolo: return # Skip during harvest
        
        faces_dir = os.path.join(self.base_dir, "faces_crop")
        os.makedirs(faces_dir, exist_ok=True)
        for i, face in enumerate(faces):
            bbox = face.bbox.astype(int)
            crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            if crop.size > 0:
                cv2.imwrite(os.path.join(faces_dir, f"{os.path.basename(image_path)}_{i}.jpg"), crop)