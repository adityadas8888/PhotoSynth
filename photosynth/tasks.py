# ~/personal/PhotoSynth/photosynth/tasks.py

from .celery_app import app
import time
import os
import socket
from .pipeline.detector import Detector
from .pipeline.captioner import Captioner

# Global instances to avoid reloading models on every task
# (Celery workers fork, so this works per worker process)
detector_instance = None
captioner_instance = None

def get_detector():
    global detector_instance
    if detector_instance is None:
        detector_instance = Detector()
    return detector_instance

def get_captioner():
    global captioner_instance
    if captioner_instance is None:
        captioner_instance = Captioner()
    return captioner_instance

# --- Task 1: Runs on 3090 PC (Detection) ---
@app.task(name='photosynth.tasks.run_detection_pass')
def run_detection_pass(file_path):
    print(f"3090 DETECT: Starting job for {file_path}")
    
    # Run Detection
    detector = get_detector()
    results = detector.run_detection(file_path)
    print(f"3090 DETECT: Results: {results}")

    # 🚨 CRITICAL: Chain the job to the VLM worker (5090 PC)
    from .tasks import run_vlm_captioning
    return run_vlm_captioning.delay(file_path)


# --- Task 2: Runs on 5090 PC (Captioning) ---
@app.task(name='photosynth.tasks.run_vlm_captioning')
def run_vlm_captioning(job_data):
    hostname = socket.gethostname()
    file_path = job_data if isinstance(job_data, str) else job_data.get('file_path')
    
    print(f"[{hostname}] VLM: Processing {file_path}")

    # Run Captioning
    captioner = get_captioner()
    caption = captioner.generate_caption(file_path)
    
    print(f"[{hostname}] VLM: Generated Caption: {caption}")

    return {"status": "SUCCESS", "file": file_path, "caption": caption, "model_used": captioner.model_type}