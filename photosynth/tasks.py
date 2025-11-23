# ~/personal/PhotoSynth/photosynth/tasks.py

from .celery_app import app
import time
import os
import socket
import hashlib
from .pipeline.detector import Detector
from .pipeline.captioner import Captioner
from .metadata import MetadataWriter
from .db import PhotoSynthDB

# Global instances (Singleton per worker)
detector_instance = None
captioner_instance = None
metadata_writer_instance = None
db_instance = None

def get_detector():
    global detector_instance
    if detector_instance is None:
        detector_instance = Detector()
    return detector_instance

def get_captioner():
    global captioner_instance
    if captioner_instance is None:
        captioner_instance = Captioner()
    return captioner_instance

def get_metadata_writer():
    global metadata_writer_instance
    if metadata_writer_instance is None:
        metadata_writer_instance = MetadataWriter()
    return metadata_writer_instance

def get_db():
    global db_instance
    if db_instance is None:
        db_instance = PhotoSynthDB()
    return db_instance

def calculate_file_hash(filepath):
    """Quick SHA256 hash for DB lookup."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        # Read first 64KB is usually enough for quick check, but full hash is safer for dedup
        # For speed in task, let's trust the watcher passed a hash or re-calc full if needed.
        # Here we re-calc full to be safe.
        for chunk in iter(lambda: f.read(65536), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

# --- Task 1: Runs on 3090 PC (Detection) ---
@app.task(name='photosynth.tasks.run_detection_pass')
def run_detection_pass(file_path):
    print(f"3090 DETECT: Starting job for {file_path}")
    
    db = get_db()
    file_hash = calculate_file_hash(file_path)
    
    # 1. Check DB: Skip if already processed
    status = db.check_status(file_hash)
    if status == 'COMPLETED':
        print(f"Skipping {file_path} (Already Processed)")
        return "SKIPPED"

    # Register/Update DB
    db.register_file(file_hash, file_path)
    db.update_status(file_hash, 'PROCESSING_DETECTION')

    # 2. Run Detection (Faces + Objects)
    detector = get_detector()
    det_results = detector.run_detection(file_path)
    
    # Save Face Data to DB (implied update, we store raw JSON for now)
    # In a real app, we'd insert into a 'faces' table.
    # For now, we pass it along to the captioning task or store in a 'face_data' column if we added it.
    # Let's assume we update the main record with face count or similar.
    
    print(f"3090 DETECT: Found {len(det_results.get('faces', []))} faces.")

    # 3. Chain to VLM (5090 PC)
    # We pass the file_path AND the detection results (so VLM knows about faces if needed)
    job_payload = {
        'file_path': file_path,
        'file_hash': file_hash,
        'det_results': det_results
    }
    
    from .tasks import run_vlm_captioning
    return run_vlm_captioning.delay(job_payload)


# --- Task 2: Runs on 5090 PC (Captioning & Metadata) ---
@app.task(name='photosynth.tasks.run_vlm_captioning')
def run_vlm_captioning(job_data):
    hostname = socket.gethostname()
    
    # Parse Payload
    if isinstance(job_data, str):
        file_path = job_data
        file_hash = calculate_file_hash(file_path) # Recalc if lost
        det_results = {}
    else:
        file_path = job_data.get('file_path')
        file_hash = job_data.get('file_hash')
        det_results = job_data.get('det_results', {})

    print(f"[{hostname}] VLM: Processing {file_path}")
    
    db = get_db()
    db.update_status(file_hash, 'PROCESSING_VLM')

    # 1. Run VLM Analysis
    captioner = get_captioner()
    analysis = captioner.generate_analysis(file_path)
    
    narrative = analysis['narrative']
    concepts = analysis['concepts']
    
    # Add detected objects to concepts if not present
    # (Simple merge)
    # if 'objects' in det_results:
    #     concepts.extend(det_results['objects'])
    
    print(f"[{hostname}] Narrative: {narrative[:50]}...")
    print(f"[{hostname}] Concepts: {concepts}")

    # 2. Write Metadata (Dual-Field Strategy)
    writer = get_metadata_writer()
    success = writer.write_metadata(file_path, narrative, concepts)

    # 3. Finalize DB
    final_status = 'COMPLETED' if success else 'ERROR_METADATA'
    db.update_status(file_hash, final_status, narrative, concepts)

    return {
        "status": final_status, 
        "file": file_path, 
        "model": captioner.model_type,
        "faces_found": len(det_results.get('faces', []))
    }