from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sqlite3
import os
import glob

app = FastAPI(title="PhotoSynth Face Tagger")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "photosynth.db")
FACES_DIR = os.path.join(BASE_DIR, "faces_crop")
UI_DIR = os.path.dirname(__file__)

# Serve Face Crops
if not os.path.exists(FACES_DIR): os.makedirs(FACES_DIR)
app.mount("/faces", StaticFiles(directory=FACES_DIR), name="faces")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# --- Models ---
class ClusterNameRequest(BaseModel):
    cluster_id: int
    name: str

# --- Endpoints ---

@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(UI_DIR, "index.html"))

@app.get("/clusters")
def get_clusters():
    """Returns grouped faces from the DB."""
    conn = get_db()
    try:
        # 1. Get all clusters (people)
        clusters = conn.execute("SELECT cluster_id, name FROM people ORDER BY cluster_id").fetchall()
        
        result = []
        for row in clusters:
            cid = row['cluster_id']
            name = row['name']
            
            # 2. Get up to 5 representative face images for this cluster
            # We join with media_files to find the crop path logic
            # Note: Our current logic saves crops as {filename}_{idx}.jpg in FACES_DIR
            # We need to map face_id back to a filename.
            
            faces_query = """
                SELECT f.face_id, m.file_path 
                FROM faces f 
                JOIN media_files m ON f.file_hash = m.file_hash 
                WHERE f.cluster_id = ? 
                LIMIT 10
            """
            face_rows = conn.execute(faces_query, (cid,)).fetchall()
            
            face_images = []
            for f_row in face_rows:
                # Heuristic: Try to find the crop file on disk
                # Crop logic was: basename + "_" + index + ".jpg"
                # Since we don't store the exact crop filename in DB (yet), we scan.
                # Better approach: Detect which crop corresponds to this face_id.
                # For now, let's find *any* crop that matches the source file.
                
                src_base = os.path.basename(f_row['file_path'])
                pattern = os.path.join(FACES_DIR, f"{src_base}_*.jpg")
                found = glob.glob(pattern)
                if found:
                    # Just take the first one found for this file
                    filename = os.path.basename(found[0])
                    face_images.append({"id": f_row['face_id'], "url": f"/faces/{filename}"})
            
            result.append({
                "id": cid,
                "name": name,
                "faces": face_images
            })
            
        return result
    finally:
        conn.close()

@app.post("/tag/cluster")
def tag_cluster(req: ClusterNameRequest):
    """Updates the name of a person."""
    conn = get_db()
    try:
        conn.execute("UPDATE people SET name = ? WHERE cluster_id = ?", (req.name, req.cluster_id))
        conn.commit()
        return {"status": "success", "message": f"Cluster {req.cluster_id} renamed to {req.name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.post("/tag/cluster")
def tag_cluster(req: ClusterNameRequest):
    conn = get_db()
    try:
        # Check if name exists elsewhere
        cursor = conn.execute(
            "SELECT cluster_id FROM people WHERE name = ? AND cluster_id != ?", 
            (req.name, req.cluster_id)
        )
        existing = cursor.fetchone()

        if existing:
            # --- MERGE LOGIC (This makes Drag & Drop work) ---
            target_id = existing[0]
            print(f"🔀 Merging Cluster {req.cluster_id} -> {target_id}")
            
            # Move faces
            conn.execute("UPDATE faces SET cluster_id = ? WHERE cluster_id = ?", (target_id, req.cluster_id))
            # Delete old cluster
            conn.execute("DELETE FROM people WHERE cluster_id = ?", (req.cluster_id,))
            
            conn.commit()
            return {"status": "merged"}
        else:
            # --- RENAME LOGIC ---
            conn.execute("UPDATE people SET name = ? WHERE cluster_id = ?", (req.name, req.cluster_id))
            conn.commit()
            return {"status": "success"}
            
    finally:
        conn.close()

@app.get("/stats")
def get_stats():
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM media_files")
    total = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM media_files WHERE status='COMPLETED'")
    processed = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM faces")
    total_faces = cursor.fetchone()[0]
    
    return {
        "total_files": total,
        "processed": processed,
        "pending": total - processed,
        "faces_found": total_faces
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

