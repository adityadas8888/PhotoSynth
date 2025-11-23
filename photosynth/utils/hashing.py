import cv2
import imagehash
import os
from PIL import Image

def calculate_content_hash(file_path):
    """
    Generates a 'Perceptual Hash' (pHash) of the visual content.
    - Ignores metadata/exif changes.
    - Stays constant even if file is modified by ExifTool.
    - Works on Images and Videos (by hashing the middle frame).
    """
    try:
        # Check file size first
        if os.path.getsize(file_path) == 0: return None

        ext = os.path.splitext(file_path)[1].lower()
        
        # --- VIDEO STRATEGY ---
        if ext in ['.mp4', '.mov', '.avi', '.mkv', '.m4v']:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened(): return None
            
            # Jump to 50% mark to avoid black start frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret: return None
            
            # Convert to PIL for hashing
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return str(imagehash.phash(img))

        # --- IMAGE STRATEGY ---
        else:
            img = Image.open(file_path)
            return str(imagehash.phash(img))
            
    except Exception as e:
        print(f"⚠️ Hashing failed for {file_path}: {e}")
        return None