#!/usr/bin/env python3
"""
Manual test script for PhotoSynth pipeline.
Processes files in ~/personal/nas/video/TEST folder.
"""

import os
import sys
from pathlib import Path
from photosynth.tasks import run_detection_pass

# Test directory
TEST_DIR = Path.home() / "personal/nas/video/TEST"

# File extensions to process
EXTENSIONS = ['.jpg', '.jpeg', '.png', '.arw', '.raw', '.tiff', '.heic', '.mp4', '.mov']

def should_skip(path):
    path_str = str(path)
    # Strict exclude
    if '@eaDir' in path_str: return True
    if '/.' in path_str: return True
    if '#recycle' in path_str: return True
    return False

def find_images(directory):
    """Find all images in directory."""
    images = []
    for ext in EXTENSIONS:
        images.extend(directory.rglob(f'*{ext}'))
        images.extend(directory.rglob(f'*{ext.upper()}'))
    
    # Filter out skipped paths
    images = [img for img in images if not should_skip(img)]
    return sorted(images)

def main():
    if not TEST_DIR.exists():
        print(f"❌ Test directory not found: {TEST_DIR}")
        print(f"   Create it with: mkdir -p {TEST_DIR}")
        sys.exit(1)
    
    images = find_images(TEST_DIR)
    
    if not images:
        print(f"No images found in {TEST_DIR}")
        print(f"Add some test images to get started!")
        sys.exit(0)
    
    print(f"📸 Found {len(images)} images in {TEST_DIR}")
    print(f"Processing...\n")
    
    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img_path.name}")
        try:
            # Trigger the pipeline
            result = run_detection_pass.delay(str(img_path))
            print(f"   ✅ Job queued: {result.id}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n✅ All jobs submitted!")
    print(f"Check logs: tail -f ~/personal/PhotoSynth/logs/worker.log")



if __name__ == "__main__":
    main()
