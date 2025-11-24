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
from .utils.faiss_manager import get_faiss_manager # <--- NEW IMPORT
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
    
    try:
        # Merge Data (Postgres JSONB returns dict, not string)
        caption_data = data['caption_data']
        if isinstance(caption_data, str):
            import json
            caption_data = json.loads(caption_data)
        
        narrative = caption_data.get('narrative', '')
        concepts = caption_data.get('concepts', [])
        
        # Optional: Add detected objects to keywords
        if data['detection_data']:
            det_data = data['detection_data']
            if isinstance(det_data, str):
                import json
                det_data = json.loads(det_data)
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
    faces_embeddings = result.get('faces', [])

    if not faces_embeddings:
        return "No faces"

    # 1. Calculate hash and register file
    safe_path = heal_path(file_path)
    file_hash = calculate_content_hash(safe_path)

    # 2. Use FAISS for Real-Time Identification
    manager = get_faiss_manager()
    db = get_db()
    db.register_file(file_hash, safe_path)  # Ensure file is registered up front

    faces_to_save = []  # Embeddings that are truly new (not found by FAISS)

    for emb in faces_embeddings:
        # Convert embedding to numpy array for FAISS search
        np_emb = np.array(emb, dtype=np.float32)

        # Search for this face in the FAISS index
        matched_id, cluster_id = manager.search_face(np_emb)

        if cluster_id is not None and cluster_id != -1:
            # Face is known! Skip saving the embedding and avoid future clustering time.
            print(f"Identified known face (Cluster: {cluster_id}). Skipping embedding save.")

        else:
            # New face: Add the embedding to the list for batch saving.
            faces_to_save.append(np_emb.tolist())  # Convert back to list for Celery serialization

    # 3. Queue the remaining NEW faces for batch DB saving
    if faces_to_save:
        # Only queue the embeddings that FAISS determined are unique
        save_faces_task.apply_async(args=[file_hash, safe_path, faces_to_save], queue='db_queue')
        return f"Found {len(faces_embeddings)} faces. {len(faces_to_save)} new queued."

    return f"Identified all {len(faces_embeddings)} faces. No new embeddings saved."


@app.task(name='photosynth.tasks.save_faces_task')
def save_faces_task(file_hash, file_path, embeddings):
    # This function remains exactly as it was, saving the NEW embeddings to the DB with cluster_id=-1
    db = get_db()
    db.register_file(file_hash, file_path)
    count = 0
    for emb in embeddings:
        import numpy as np
        arr = np.array(emb, dtype=np.float32)
        db.add_face(file_hash, arr)
        count += 1
    print(f"💾 DB Saved: {count} new faces.")
    return count


# In photosynth/tasks.py (Add this new function)

@app.task(name='photosynth.tasks.run_clustering_task')
def run_clustering_task(total_embeddings_count):
    print(f"🧠 STARTING CLUSTERING of {total_embeddings_count} embeddings...")
    db = get_db()

    # 1. Load Data
    all_face_data = db.get_all_embeddings()
    embeddings = np.array([d[1] for d in all_face_data], dtype=np.float32)
    face_ids = [d[0] for d in all_face_data]

    d = embeddings.shape[1]

    # 2. Define FAISS K-Means Clusterer
    # Use a large number of clusters (K) and a relatively small number of iterations (niter)
    # The Index is built on the GPU for speed.
    k = 10000
    niter = 25

    console.print(f"Running FAISS K-Means with k={k} on {len(embeddings)} vectors...")

    # Transfer data to GPU for clustering if available
    if faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        # Initialize K-Means on the GPU
        kmeans = faiss.GpuClustering(d, k, res=res)
    else:
        # Fallback to CPU clustering
        kmeans = faiss.Clustering(d, k)

    kmeans.niter = niter
    kmeans.max_points_per_centroid = 1000000  # Allow large clusters

    # 3. Train the K-Means Model
    kmeans.train(embeddings)

    # 4. Assign Clusters (Find which cluster each vector belongs to)
    D, I = kmeans.index.search(embeddings, 1)

    # 5. Prepare Batch Update for DB
    cluster_map = []
    for i, cluster_index in enumerate(I):
        # We assign the new cluster index as the new cluster_id
        cluster_id = int(cluster_index[0])
        face_id = face_ids[i]
        cluster_map.append((cluster_id, face_id))

    # 6. Update DB
    db.update_clusters(cluster_map)

    # 7. Rebuild FAISS Index after Clustering
    # This ensures the FAISS search logic (in extract_faces_task) is using the updated clusters
    manager = get_faiss_manager()
    manager.index = None  # Force a rebuild
    manager.build_index_if_missing()

    return f"Clustered {len(embeddings)} faces into {kmeans.obj[2]} clusters."