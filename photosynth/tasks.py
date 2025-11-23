from .celery_app import app
import os
import numpy as np
from .pipeline.detector import Detector
from .pipeline.captioner import Captioner
from .metadata import MetadataWriter
from .db import PhotoSynthDB
from .utils.hashing import calculate_content_hash # <--- NEW IMPORT
from .utils.paths import heal_path

# Singletons
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

# --- DAILY PIPELINE ---

@app.task(name='photosynth.tasks.run_detection_pass')
def run_detection_pass(file_path):
    print(f"🔍 DAILY DETECT: {os.path.basename(file_path)}")
    
    db = get_db()
    # Use Visual Hash
    file_hash = calculate_content_hash(file_path)
    
    if not file_hash: return "ERROR_HASH"
    
    if db.check_status(file_hash) == 'COMPLETED':
        print(f"Skipping {file_path} (Done)")
        return "SKIPPED"

    db.register_file(file_hash, file_path)
    db.update_status(file_hash, 'PROCESSING_DETECTION')

    detector = get_detector()
    det_results = detector.run_detection(file_path)

    job_payload = {
        'file_path': file_path,
        'file_hash': file_hash,
        'det_results': det_results
    }
    
    # Forward to 5090
    run_vlm_captioning.apply_async(args=[job_payload], queue='vlm_queue')
    return f"Detected {len(det_results.get('objects', []))} objects"

@app.task(name='photosynth.tasks.run_vlm_captioning')
def run_vlm_captioning(job_data):
    if isinstance(job_data, str):
        file_path = job_data
        file_hash = calculate_content_hash(file_path)
        det_results = {}
    else:
        file_path = job_data.get('file_path')
        file_hash = job_data.get('file_hash')
        det_results = job_data.get('det_results', {})

    print(f"🤖 VLM CAPTION: {os.path.basename(file_path)}")
    db = get_db()
    db.update_status(file_hash, 'PROCESSING_VLM')

    captioner = get_captioner()
    analysis = captioner.generate_analysis(file_path, det_results)
    
    # Keyword Validation
    if not analysis['concepts']:
        print(f"⚠️ WARNING: No keywords generated for {os.path.basename(file_path)}. Adding fallback.")
        analysis['concepts'] = ["needs_review"]
    
    writer = get_writer()
    success = writer.write_metadata(file_path, analysis['narrative'], analysis['concepts'])

    final_status = 'COMPLETED' if success else 'ERROR_METADATA'
    db.update_status(file_hash, final_status, analysis['narrative'], analysis['concepts'])

    return {"status": final_status, "file": file_path}

# --- HARVEST TASKS ---

@app.task(name='photosynth.tasks.extract_faces_task')
def extract_faces_task(file_path):
    detector = get_detector()
    result = detector._process_image(file_path)
    faces = result.get('faces', [])
    
    if faces:
        # Use path healing for hash calc to ensure consistency
        safe_path = heal_path(file_path)
        file_hash = calculate_content_hash(safe_path)
        
        save_faces_task.apply_async(args=[file_hash, file_path, faces], queue='db_queue')
        return f"Found {len(faces)} faces"
    return "No faces"

@app.task(name='photosynth.tasks.save_faces_task')
def save_faces_task(file_hash, file_path, embeddings):
    db = get_db()
    db.register_file(file_hash, file_path)
    count = 0
    for emb in embeddings:
        import numpy as np
        arr = np.array(emb, dtype=np.float32)
        db.add_face(file_hash, arr)
        count += 1
    print(f"💾 DB Saved: {count} faces")
    return count