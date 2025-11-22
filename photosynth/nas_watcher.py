import time
import hashlib
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path
from .tasks import run_detection_pass  # Import the starting Celery task

import yaml

# Load Configuration
SETTINGS_PATH = os.path.join(os.path.dirname(__file__), '../settings.yaml')
with open(SETTINGS_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Your mounted NAS photo shares (the ones to watch)
nas_mount = Path(config['paths']['nas_mount']).expanduser()
WATCH_DIRS = [nas_mount / d for d in config['paths']['watch_dirs']]

# File extensions to process (photos and common video formats)
FILE_PATTERNS = [
    '.jpg', '.jpeg', '.png', '.arw', '.raw', '.tiff',  # Photo
    '.mp4', '.mov', '.avi', '.mkv'  # Video
]


import imagehash
from PIL import Image

# --- Hashing Utility ---
def calculate_phash(filepath):
    """Generates a Perceptual Hash (pHash) which is robust to metadata changes."""
    try:
        img = Image.open(filepath)
        # 8x8 pHash is standard (64 bits)
        phash = str(imagehash.phash(img))
        return phash
    except Exception as e:
        print(f"⚠️ Could not calculate pHash for {filepath}: {e}")
        # Fallback to filename+size if image is unreadable (e.g. video)
        # For videos, pHash is harder. We might need a different strategy or just use path for now.
        return f"ERR_{os.path.basename(filepath)}"


# --- Watchdog Event Handler ---
class PhotoSynthHandler(FileSystemEventHandler):
    @staticmethod
    def on_created(event):
        """Called when a new file or directory is created."""
        if event.is_directory:
            return

        src_path = event.src_path
        
        # Skip Synology metadata directories
        if '@eaDir' in src_path:
            return
        
        file_ext = Path(src_path).suffix.lower()

        if file_ext in FILE_PATTERNS:
            print(f"[{time.strftime('%H:%M:%S')}] NEW FILE DETECTED: {src_path}")

            # CRITICAL STEP: Wait for file transfer to complete before hashing/processing
            time.sleep(1)

            try:
                # 1. Generate unique hash for tracking (pHash)
                file_hash = calculate_phash(src_path)

                # 2. Inject job into the Detection Queue (starts the pipeline)
                # The task is routed to the 3090 PC worker via 'detection_queue'
                run_detection_pass.delay(src_path)

                print(f"Job queued successfully. Hash: {file_hash}...")

            except FileNotFoundError:
                print(f"Error: File disappeared before processing: {src_path}")
            except Exception as e:
                print(f"Error processing file {src_path}: {e}")


# --- Main Watcher Loop ---
def start_watcher():
    event_handler = PhotoSynthHandler()
    observer = Observer()

    for path_str in WATCH_DIRS:
        # Schedule monitoring for all defined paths, recursively
        path = str(path_str.resolve())
        observer.schedule(event_handler, path, recursive=True)
        print(f"Monitoring: {path}")

    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    start_watcher()