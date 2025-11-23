import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path
from .tasks import run_detection_pass
import yaml

# Load Config
SETTINGS_PATH = os.path.join(os.path.dirname(__file__), '../settings.yaml')
with open(SETTINGS_PATH, 'r') as f:
    config = yaml.safe_load(f)

nas_mount = Path(config['paths']['nas_mount']).expanduser()
WATCH_DIRS = [nas_mount / d for d in config['paths']['watch_dirs']]

FILE_PATTERNS = ['.jpg', '.jpeg', '.png', '.arw', '.raw', '.tiff', '.mp4', '.mov', '.avi', '.mkv']

class PhotoSynthHandler(FileSystemEventHandler):
    @staticmethod
    def is_safe_path(path_str):
        """Strictly exclude Synology internal folders and hidden files."""
        # 1. Exclude the dreaded @eaDir
        if '@eaDir' in path_str:
            return False
        
        # 2. Exclude hidden files (Mac/Linux)
        if '/.' in path_str:
            return False
            
        # 3. Exclude recycle bins
        if '#recycle' in path_str:
            return False
            
        return True

    def on_created(self, event):
        if event.is_directory: return

        src_path = event.src_path
        
        # --- THE FIX ---
        if not self.is_safe_path(src_path):
            return
        # ---------------

        file_ext = Path(src_path).suffix.lower()
        if file_ext in FILE_PATTERNS:
            print(f"[{time.strftime('%H:%M:%S')}] 📸 Detected: {os.path.basename(src_path)}")
            
            # Wait for file write to settle (prevents reading partial files)
            time.sleep(2)
            
            try:
                # Trigger Pipeline
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