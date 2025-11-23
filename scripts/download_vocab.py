#!/usr/bin/env python3
import requests
import yaml
import json
import os

# Official Ultralytics LVIS configuration (Stable)
URL = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/lvis.yaml"

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(BASE_DIR, "photosynth", "vocabulary.json")

# Your Custom "Life" Tags (Things LVIS might miss or you want to emphasize)
CUSTOM_TAGS = [
    "face", "selfie", "crowd",
    "bmw", "tesla", "sportscar", "electric car","toyota"
    "indian food", "curry", "sari", "kurta", "lehenga","saree"
    "pc gaming", "rgb lighting", "graphics card", "server",
    "sunset", "sunrise", "hiking trail", "waterfall", "lego",
    "receipt", "invoice", "document", "screen","license plate","screenshot"
]

def main():
    print(f"⬇️  Downloading LVIS config from Ultralytics...")
    try:
        resp = requests.get(URL)
        resp.raise_for_status()
        
        # Parse YAML
        data = yaml.safe_load(resp.text)
        
        # Extract names (it's a dict {0: 'name', 1: 'name'...})
        lvis_classes = list(data['names'].values())
        
        # Clean & Merge
        # 1. Lowercase everything
        lvis_classes = [c.lower() for c in lvis_classes]
        # 2. Merge with custom tags
        full_vocab = sorted(list(set(lvis_classes + CUSTOM_TAGS)))
        
        print(f"✅ Parsed {len(lvis_classes)} classes from LVIS.")
        print(f"➕ Added {len(CUSTOM_TAGS)} custom tags.")
        print(f"🎉 Total Vocabulary: {len(full_vocab)} classes.")
        
        # Save to JSON
        with open(OUTPUT_FILE, "w") as f:
            json.dump(full_vocab, f, indent=2)
            
        print(f"💾 Saved to {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()