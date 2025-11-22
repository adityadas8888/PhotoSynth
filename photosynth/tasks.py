# ~/personal/PhotoSynth/photosynth/tasks.py

from .celery_app import app
import time  # Import for simulation
from .celery_app import app
import torch
import os
import socket

# --- Task 1: Runs on 3090 PC (Detection) ---
# Note: The 'name' matches the routing defined in celery_app.py
@app.task(name='photosynth.tasks.run_detection_pass')
def run_detection_pass(file_path):
    # In a real pipeline, we'd load the Pydantic Job Schema here,
    # but for testing, we just use the path.
    print(f"3090 DETECT: Starting job for {file_path}")
    time.sleep(2)  # Simulate heavy detection work

    # 🚨 CRITICAL: Chain the job to the VLM worker (5090 PC)
    # This automatically puts the task into the 'vlm_queue'
    from .tasks import run_vlm_captioning
    return run_vlm_captioning.delay(file_path)


# --- Task 2: Runs on 5090 PC (Captioning) ---
@app.task(name='photosynth.tasks.run_vlm_captioning')
def run_vlm_captioning(file_path):
    print(f"5090 VLM: Starting job for {file_path}")
    time.sleep(5)  # Simulate heavy VLM work

    # In a real pipeline, ExifTool commitment would happen here.
    return f"COMPLETED: Metadata added to {file_path}"


@app.task(name='photosynth.tasks.run_vlm_captioning')
def run_vlm_captioning(job_data):
    hostname = socket.gethostname()
    file_path = job_data if isinstance(job_data, str) else job_data.get('file_path')

    print(f"[{hostname}] VLM: Starting job for {file_path}")

    # 🤖 LOGIC: Choose Model based on Hostname
    if "5090" in hostname:
        # --- RTX 5090: Run Qwen2-VL (The Backlog Beast) ---
        print("🚀 Mode: RTX 5090 (Backlog) -> Loading Qwen2-VL")
        # Placeholder for actual Qwen loading logic
        # from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        # model = Qwen2VLForConditionalGeneration.from_pretrained("models/qwen2_vl", device_map="auto")
        # ... inference code ...
        caption = f"[Qwen] Processed {os.path.basename(file_path)}"

    else:
        # --- RTX 3090: Run Llama 3.2 (Day-to-Day) ---
        print("🌿 Mode: RTX 3090 (Daily) -> Loading Llama 3.2 Vision (4-bit)")
        # Placeholder for actual Llama loading logic (using bitsandbytes for 4-bit)
        # from transformers import MllamaForConditionalGeneration, AutoProcessor
        # model = MllamaForConditionalGeneration.from_pretrained("models/llama_3_2_vision", load_in_4bit=True)
        # ... inference code ...
        caption = f"[Llama] Processed {os.path.basename(file_path)}"

    return {"status": "SUCCESS", "file": file_path, "caption": caption}