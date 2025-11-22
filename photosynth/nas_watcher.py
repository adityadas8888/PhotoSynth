import time
import hashlib
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path
from .tasks import run_detection_pass  # Import the starting Celery task

# --- Configuration ---
# Your mounted NAS photo shares (the ones to watch)
WATCH_DIRS = [
    Path("~/personal/nas/photo").expanduser(),
    Path("~/personal/nas/video").expanduser(),
    # Path("~/personal/nas/homes").expanduser(), # Uncomment if you want to watch homes too
]

# File extensions to process (photos and common video formats)
FILE_PATTERNS = [
    '.jpg', '.jpeg', '.png', '.arw', '.raw', '.tiff',  # Photo
    '.mp4', '.mov', '.avi', '.mkv'  # Video
]


# --- Hashing Utility ---
def hash_file_sha256(filepath):
    """Generates a SHA256 hash for a file in chunks to handle large files efficiently."""
    hasher = hashlib.sha256()
    # Read the file in 64KB chunks
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


# --- Watchdog Event Handler ---
class PhotoSynthHandler(FileSystemEventHandler):
    @staticmethod
    def on_created(event):
        """Called when a new file or directory is created."""
        if event.is_directory:
            return

        src_path = event.src_path
        file_ext = Path(src_path).suffix.lower()

        if file_ext in FILE_PATTERNS:
            print(f"[{time.strftime('%H:%M:%S')}] NEW FILE DETECTED: {src_path}")

            # CRITICAL STEP: Wait for file transfer to complete before hashing/processing
            time.sleep(1)

            try:
                # 1. Generate unique hash for tracking
                file_hash = hash_file_sha256(src_path)

                # 2. Inject job into the Detection Queue (starts the pipeline)
                # The task is routed to the 3090 PC worker via 'detection_queue'
                run_detection_pass.delay(src_path)

                print(f"Job queued successfully. Hash: {file_hash[:8]}...")

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