# ~/personal/PhotoSynth/photosynth/tasks.py

from .celery_app import app
import time  # Import for simulation


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