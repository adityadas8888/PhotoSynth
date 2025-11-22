# ~/personal/PhotoSynth/photosynth/celery_app.py

from celery import Celery

# Broker and backend point to the Redis instance running on 10.0.0.230
# If your 3090 PC IP changes, you MUST update this line!
REDIS_BROKER_URL = 'redis://10.10.10.2:6379/0'

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
    }
)