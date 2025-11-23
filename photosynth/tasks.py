from .celery_app import app
import time
from .pipeline.detector import Detector
from .pipeline.captioner import Captioner
from .metadata import MetadataWriter
from .db import PhotoSynthDB
import hashlib

# Singleton Getters
detector_instance = None
captioner_instance = None
writer_instance = None
db_instance = None

def get_detector():
    global detector_instance
    if detector_instance is None: detector_instance = Detector()
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
    with open(filepath, 'rb') as f:
        # Read 64KB for speed
        hasher.update(f.read(65536))
    return hasher.hexdigest()

@app.task(name='photosynth.tasks.run_detection_pass')
def run_detection_pass(file_path):
    print(f"🔍 DETECT: {file_path}")
    db = get_db()
    file_hash = calculate_file_hash(file_path)
    
    if db.check_status(file_hash) == 'COMPLETED':
        print(f"Skipping {file_path}")
        return "SKIPPED"

    db.register_file(file_hash, file_path)
    db.update_status(file_hash, 'PROCESSING_DETECTION')

    detector = get_detector()
    det_results = detector.run_detection(file_path)

    # Chain to VLM
    job_payload = {'file_path': file_path, 'file_hash': file_hash, 'det_results': det_results}
    from .tasks import run_vlm_captioning
    return run_vlm_captioning.delay(job_payload)

@app.task(name='photosynth.tasks.run_vlm_captioning')
def run_vlm_captioning(job_data):
    if isinstance(job_data, str):
        file_path = job_data
        det_results = {}
        file_hash = calculate_file_hash(file_path)
    else:
        file_path = job_data.get('file_path')
        file_hash = job_data.get('file_hash')
        det_results = job_data.get('det_results', {})

    print(f"🤖 VLM: {file_path}")
    db = get_db()
    db.update_status(file_hash, 'PROCESSING_VLM')

    captioner = get_captioner()
    # Pass detection context (Florence objects) to Llama
    analysis = captioner.generate_analysis(file_path, det_results)
    
    narrative = analysis['narrative']
    concepts = analysis['concepts']

    # Write Metadata (Strict No-Rename)
    writer = get_writer()
    success = writer.write_metadata(file_path, narrative, concepts)

    final_status = 'COMPLETED' if success else 'ERROR_METADATA'
    db.update_status(file_hash, final_status, narrative, concepts)

    return {"status": final_status, "file": file_path, "tags": concepts}