import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path
from .tasks import run_detection_pass
from .db import PhotoSynthDB
from .utils.hashing import calculate_content_hash
import yaml

# Load Config
SETTINGS_PATH = os.path.join(os.path.dirname(__file__), '../settings.yaml')
with open(SETTINGS_PATH, 'r') as f:
    config = yaml.safe_load(f)

nas_mount = Path(config['paths']['nas_mount']).expanduser()
WATCH_DIRS = [nas_mount / d for d in config['paths']['watch_dirs']]
FILE_PATTERNS = ['.jpg', '.jpeg', '.png', '.arw', '.mp4', '.mov', '.mkv']

class PhotoSynthHandler(FileSystemEventHandler):
    def __init__(self):
        self.db = PhotoSynthDB()

    @staticmethod
    def is_safe_path(path_str):
        if '@eaDir' in path_str: return False
        if '/.' in path_str: return False
        if '#recycle' in path_str: return False
        return True

    def on_modified(self, event):
        # Synology often triggers 'modified' instead of 'created' for uploads
        self.process(event.src_path)

    def on_created(self, event):
        self.process(event.src_path)

    def process(self, src_path):
        if os.path.isdir(src_path): return
        if not self.is_safe_path(src_path): return

        file_ext = Path(src_path).suffix.lower()
        if file_ext in FILE_PATTERNS:
            # Debounce: Wait for write to finish
            time.sleep(2)
            
            try:
                # 1. Calculate Visual Hash (Ignores Metadata Changes)
                f_hash = calculate_content_hash(src_path)
                if not f_hash: return

                # 2. Check DB immediately
                # If we just finished processing this file, the hash matches the DB record.
                status = self.db.check_status(f_hash)
                
                if status == 'COMPLETED':
                    print(f"💤 Ignoring metadata update: {os.path.basename(src_path)}")
                    return
                
                if status == 'PROCESSING_DETECTION' or status == 'PROCESSING_VLM':
                    # Already in pipeline, ignore duplicate event
                    return

                # 3. Valid New Content -> Queue It
                print(f"📸 New Content Detected: {os.path.basename(src_path)}")
                run_detection_pass.delay(src_path)

            except Exception as e:
                print(f"Error queuing {src_path}: {e}")

def start_watcher():
    event_handler = PhotoSynthHandler()
    observer = Observer()

    for path_str in WATCH_DIRS:
        if not path_str.exists():
            print(f"⚠️ Warning: Watch dir does not exist: {path_str}")
            continue
        print(f"👀 Watching: {path_str}")
        observer.schedule(event_handler, str(path_str), recursive=True)

    observer.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_watcher()