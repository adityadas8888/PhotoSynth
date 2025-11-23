from .celery_app import app
import os
import hashlib
import numpy as np
from .pipeline.detector import Detector
from .pipeline.captioner import Captioner
from .metadata import MetadataWriter
from .db import PhotoSynthDB

# --- Singletons ---
detector_instance = None
captioner_instance = None
writer_instance = None
db_instance = None

def get_detector():
    global detector_instance
    if detector_instance is None: detector_instance = Detector(enable_yolo=True)
    return detector_instance

def get_captioner():
    global captioner_instance
    if captioner_instance is None: captioner_instance = Captioner()
    return captioner_instance

def get_writer():
    global writer_instance
    if writer_instance is None: writer_instance = MetadataWriter()
    return writer_instance

def get_db():
    global db_instance
    if db_instance is None: db_instance = PhotoSynthDB()
    return db_instance

def calculate_file_hash(filepath):
    hasher = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            hasher.update(f.read(65536))
        return hasher.hexdigest()
    except: return None

# =========================================================================
#  PART 1: DAILY PIPELINE (Watcher -> Detect -> VLM -> Metadata)
# =========================================================================

@app.task(name='photosynth.tasks.run_detection_pass')
def run_detection_pass(file_path):
    print(f"🔍 DAILY DETECT: {os.path.basename(file_path)}")
    
    db = get_db()
    file_hash = calculate_file_hash(file_path)
    
    if not file_hash: return "ERROR_HASH"
    
    # Check duplicates
    if db.check_status(file_hash) == 'COMPLETED':
        print(f"Skipping {file_path} (Done)")
        return "SKIPPED"

    db.register_file(file_hash, file_path)
    db.update_status(file_hash, 'PROCESSING_DETECTION')

    # Run Master Detector (Face + ID + YOLO)
    detector = get_detector()
    det_results = detector.run_detection(file_path)

    # Chain to VLM (Send to vlm_queue so 5090 picks it up)
    job_payload = {
        'file_path': file_path,
        'file_hash': file_hash,
        'det_results': det_results
    }
    
    run_vlm_captioning.apply_async(args=[job_payload], queue='vlm_queue')
    return f"Detected {len(det_results.get('objects', []))} objects"

@app.task(name='photosynth.tasks.run_vlm_captioning')
def run_vlm_captioning(job_data):
    # Unpack
    if isinstance(job_data, str):
        file_path = job_data
        file_hash = calculate_file_hash(file_path)
        det_results = {}
    else:
        file_path = job_data.get('file_path')
        file_hash = job_data.get('file_hash')
        det_results = job_data.get('det_results', {})

    print(f"🤖 VLM CAPTION: {os.path.basename(file_path)}")
    db = get_db()
    db.update_status(file_hash, 'PROCESSING_VLM')

    # 1. Generate Caption & Tags
    captioner = get_captioner()
    analysis = captioner.generate_analysis(file_path, det_results)
    
    narrative = analysis['narrative']
    concepts = analysis['concepts']

    # 2. Write Metadata (Using Path Healing internally)
    writer = get_writer()
    success = writer.write_metadata(file_path, narrative, concepts)

    # 3. Save to DB
    final_status = 'COMPLETED' if success else 'ERROR_METADATA'
    db.update_status(file_hash, final_status, narrative, concepts)

    return {"status": final_status, "file": file_path}

# =========================================================================
#  PART 2: HARVEST PIPELINE (Scan Faces -> Save DB)
# =========================================================================

@app.task(name='photosynth.tasks.extract_faces_task')
def extract_faces_task(file_path):
    # Runs on GPU Workers (3090 & 5090)
    print(f"👤 HARVEST SCAN: {os.path.basename(file_path)}")
    
    detector = get_detector()
    # Use internal method to just get embeddings (Skip YOLO)
    result = detector._process_image(file_path) 
    
    faces = result.get('faces', [])
    
    if faces:
        file_hash = calculate_file_hash(detector._heal_path(file_path))
        # Send to DB Worker (3090 only)
        save_faces_task.apply_async(args=[file_hash, file_path, faces], queue='db_queue')
        return f"Found {len(faces)} faces"
    return "No faces"

@app.task(name='photosynth.tasks.save_faces_task')
def save_faces_task(file_hash, file_path, embeddings):
    # Runs on DB Worker (3090)
    db = get_db()
    db.register_file(file_hash, file_path)
    
    count = 0
    for emb in embeddings:
        arr = np.array(emb, dtype=np.float32)
        db.add_face(file_hash, arr)
        count += 1
        
    print(f"💾 DB Saved: {count} faces")
    return count