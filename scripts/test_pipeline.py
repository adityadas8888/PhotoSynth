#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from photosynth.tasks import run_detection_pass

# Config
TEST_DIR = Path(os.path.expanduser("~/personal/nas/video/TEST"))
EXTENSIONS = ['.jpg', '.jpeg', '.png', '.mp4', '.mov']

def should_skip(path):
    """Strict exclusions for Synology/System files."""
    p = str(path)
    if '@eaDir' in p: return True
    if '/.' in p: return True       # Hidden files .DS_Store
    if '#recycle' in p: return True
    return False

def main():
    if not TEST_DIR.exists():
        print(f"❌ Directory not found: {TEST_DIR}")
        sys.exit(1)

    files = []
    for ext in EXTENSIONS:
        files.extend(TEST_DIR.rglob(f"*{ext}"))
        files.extend(TEST_DIR.rglob(f"*{ext.upper()}"))
    
    # Filter
    files = [f for f in files if not should_skip(f)]
    files.sort()

    print(f"📸 Found {len(files)} test files.")
    print("🚀 Injecting into Daily Pipeline...")

    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Queuing: {f.name}")
        
        # This triggers the full chain: 3090 Detect -> 5090 Caption
        run_detection_pass.delay(str(f))

    print("\n✅ Jobs submitted! Watch logs on both machines.")

if __name__ == "__main__":
    main()