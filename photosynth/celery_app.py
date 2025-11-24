# No changes needed to code logic based on analysis, but I will use this tool to verify I can write to the file if needed, or just skip.
# Actually, I will just proceed with commands.

from celery import Celery

# Broker and backend point to the Redis instance running on 10.0.0.230
# If your 3090 PC IP changes, you MUST update this line!
import yaml
import os

# Load Configuration
SETTINGS_PATH = os.path.join(os.path.dirname(__file__), '../settings.yaml')
with open(SETTINGS_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Broker and backend point to the Redis instance defined in settings
import socket

hostname = socket.gethostname()
if "5090" in hostname:
    # Node B (Worker) connects to Node A
    REDIS_BROKER_URL = "redis://10.0.0.230:6379/0"
else:
    # Node A (Host) connects to itself
    REDIS_BROKER_URL = "redis://127.0.0.1:6379/0"

app = Celery(
    'PhotoSynth',
    broker=REDIS_BROKER_URL,
    backend=REDIS_BROKER_URL
)

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='America/Los_Angeles', # Using a standard US timezone
    enable_utc=True,
    # 🚨 Optional: Define specific queues for explicit routing
    task_routes = {
        'photosynth.tasks.run_detection_pass': {'queue': 'detection_queue'},
        'photosynth.tasks.run_vlm_captioning': {'queue': 'vlm_queue'},
        'photosynth.tasks.finalize_file': {'queue': 'detection_queue'},
        'photosynth.tasks.save_faces_task': {'queue': 'detection_queue'},
    }
)