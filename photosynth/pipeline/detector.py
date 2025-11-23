import cv2
import os
import torch
import json
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from ultralytics import YOLOWorld

class Detector:
    def __init__(self, enable_yolo=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enable_yolo = enable_yolo
        self.face_app = None
        self.yolo_model = None
        
        # Paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.models_dir = os.path.join(self.base_dir, "models")
        self.vocab_path = os.path.join(self.base_dir, "photosynth", "vocabulary.json")
        
        self._load_models()

    def _load_models(self):
        # 1. InsightFace (Always needed)
        print(f"[{self.device}] 🔍 Loading InsightFace...")
        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        # 2. YOLO-World (Only if enabled)
        if self.enable_yolo:
            print(f"[{self.device}] 🦅 Loading YOLO-World...")
            # Check local path first
            local_yolo = os.path.join(self.models_dir, 'yolov8l-worldv2.pt')
            if os.path.exists(local_yolo):
                self.yolo_model = YOLOWorld(local_yolo)
            else:
                self.yolo_model = YOLOWorld('yolov8l-worldv2.pt')
                
            self.yolo_model.to(self.device)
            
            # FIX: Ensure this runs ONLY if YOLO is enabled
            if os.path.exists(self.vocab_path):
                with open(self.vocab_path, 'r') as f:
                    vocab = json.load(f)
                self.yolo_model.set_classes(vocab)
                print(f"   📚 YOLO Loaded with {len(vocab)} classes")

    def _heal_path(self, file_path):
        """Fixes path mismatch between computers."""
        if os.path.exists(file_path): return file_path
        if "personal/nas" in file_path:
            relative = file_path.split("personal/nas")[-1]
            new_path = os.path.join(os.path.expanduser("~"), "personal/nas", relative.strip("/"))
            if os.path.exists(new_path): return new_path
        return file_path

    def _identify_faces(self, faces):
        """Compares detected faces against the Cluster DB."""
        from photosynth.db import PhotoSynthDB
        try:
            db = PhotoSynthDB()
            known_faces = db.get_known_faces() 
        except Exception:
            return [] 

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
        """Entry point for Daily Operations."""
        print(f"Processing {os.path.basename(file_path)}...")
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.mp4', '.mov', '.avi', '.mkv', '.m4v']:
            return self._process_video(file_path)
        else:
            return self._process_image(file_path)

    def _process_image(self, image_path):
        image_path = self._heal_path(image_path)
        image_cv = cv2.imread(image_path)
        if image_cv is None: return {}
        
        # 1. Face Detect
        faces = self.face_app.get(image_cv)
        # self._save_face_crops(faces, image_cv, image_path) # Optional: Enable if you want crops for every file
        
        # 2. Identify People
        known_people = self._identify_faces(faces)
        
        # 3. YOLO Detect (Only if enabled)
        objs = []
        if self.enable_yolo:
            results = self.yolo_model.predict(image_path, conf=0.05, verbose=False)
            for r in results:
                for c in r.boxes.cls:
                    objs.append(self.yolo_model.names[int(c)])
        
        return {
            "status": "SUCCESS",
            "faces": [f.embedding.tolist() for f in faces], 
            "face_count": len(faces),
            "known_people": known_people,
            "objects": list(set(objs)),
            "is_video": False
        }

    def _process_video(self, video_path):
        print(f"🎬 Video detected. Sampling...")
        video_path = self._heal_path(video_path)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return {"status": "ERROR", "faces": [], "objects": [], "known_people": []}

        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = raw_fps if raw_fps > 0 else 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        if duration > 30: interval_sec = 5
        elif duration > 5: interval_sec = 2
        else: interval_sec = 1
            
        frame_interval = int(fps * interval_sec)
        if frame_interval < 1: frame_interval = 1
        
        print(f"   Duration: {duration:.1f}s -> Sampling every {interval_sec}s")
        
        all_objects = set()
        all_people = set()
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_idx % frame_interval == 0:
                faces = self.face_app.get(frame)
                names = self._identify_faces(faces)
                all_people.update(names)
                
                if self.enable_yolo:
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
            "known_people": list(all_people),
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