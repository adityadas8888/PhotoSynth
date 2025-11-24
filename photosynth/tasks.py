from .celery_app import app
import os
import torch
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
    file_hash = calculate_content_hash(file_path)
    
    if not file_hash: return "ERROR_HASH"
    
    # Check if already done
    data = db.get_file_data(file_hash)
    if data and data.get('detection_status') == 'COMPLETED':
        return "SKIPPED_DONE"

    db.register_file(file_hash, file_path)
    db.update_detection_result(file_hash, 'PROCESSING')

    detector = get_detector()
    det_results = detector.run_detection(file_path)

    # Save Results
    db.update_detection_result(file_hash, 'COMPLETED', det_results)
    
    # Check if we can finalize (if captioning is already done)
    # Re-fetch to get latest status
    data = db.get_file_data(file_hash)
    if data.get('caption_status') == 'COMPLETED':
        finalize_file.delay(file_hash)
        
    return f"Detected {len(det_results.get('objects', []))} objects"

@app.task(name='photosynth.tasks.run_vlm_captioning')
def run_vlm_captioning(file_path):
    # Heal path for 5090 context
    file_path = heal_path(file_path)
    print(f"🤖 VLM CAPTION: {os.path.basename(file_path)}")
    
    db = get_db()
    file_hash = calculate_content_hash(file_path)
    
    # Check if already done
    data = db.get_file_data(file_hash)
    if data and data.get('caption_status') == 'COMPLETED':
        return "SKIPPED_DONE"

    # Fetch detection context if available
    import json
    det_results = {}
    if data and data.get('detection_data'):
        try:
            det_results = json.loads(data['detection_data'])
        except: pass

    db.update_caption_result(file_hash, 'PROCESSING')

    # CRITICAL: Free VRAM by unloading detector before loading VLM
    global detector_instance
    if detector_instance is not None:
        print("🧹 Unloading Detector to free VRAM...")
        del detector_instance
        detector_instance = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    
    captioner = get_captioner()
    analysis = captioner.generate_analysis(file_path, det_results)
    
    # Keyword Validation
    if not analysis['concepts']:
        print(f"⚠️ WARNING: No keywords generated for {os.path.basename(file_path)}. Adding fallback.")
        analysis['concepts'] = ["needs_review"]
    
    # Save Results
    db.update_caption_result(file_hash, 'COMPLETED', analysis)

    # Check if we can finalize (if detection is already done)
    # Re-fetch to get latest status
    data = db.get_file_data(file_hash)
    if data.get('detection_status') == 'COMPLETED':
        finalize_file.delay(file_hash)

    return {"status": "COMPLETED", "file": file_path}

@app.task(name='photosynth.tasks.finalize_file')
def finalize_file(file_hash):
    print(f"🏁 FINALIZING: {file_hash}")
    db = get_db()
    data = db.get_file_data(file_hash)
    
    if not data: return "ERROR_NO_DATA"
    if data['status'] == 'COMPLETED': return "ALREADY_COMPLETED"
    
    import json
    try:
        # Merge Data
        caption_data = json.loads(data['caption_data'])
        
        narrative = caption_data.get('narrative', '')
        concepts = caption_data.get('concepts', [])
        
        # Optional: Add detected objects to keywords
        if data['detection_data']:
            det_data = json.loads(data['detection_data'])
            objects = det_data.get('objects', [])
            concepts.extend(objects)
            concepts = list(set(concepts)) # Deduplicate
            
        file_path = heal_path(data['file_path'])
        
        writer = get_writer()
        success = writer.write_metadata(file_path, narrative, concepts)
        
        final_status = 'COMPLETED' if success else 'ERROR_METADATA'
        db.update_status(file_hash, final_status, narrative, concepts)
        return final_status
        
    except Exception as e:
        print(f"❌ Finalization Error: {e}")
        return "ERROR_EXCEPTION"

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