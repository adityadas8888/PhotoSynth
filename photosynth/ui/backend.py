from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sqlite3
import os
import json

app = FastAPI(title="PhotoSynth Face Tagger")

# CORS (Allow frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Path
DB_PATH = os.path.expanduser("~/personal/PhotoSynth/photosynth.db")
FACES_DIR = os.path.expanduser("~/personal/PhotoSynth/faces_crop")
UI_DIR = os.path.dirname(__file__)

# Serve Face Crops Static Files
if not os.path.exists(FACES_DIR):
    os.makedirs(FACES_DIR)
app.mount("/faces", StaticFiles(directory=FACES_DIR), name="faces")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# --- API Models ---
class TagRequest(BaseModel):
    face_id: str
    name: str

class ClusterRequest(BaseModel):
    cluster_id: int
    name: str

# --- Endpoints ---

@app.get("/")
def serve_ui():
    """Serves the main UI."""
    return FileResponse(os.path.join(UI_DIR, "index.html"))

@app.get("/clusters")
def get_clusters():
    """
    Returns grouped faces.
    For now, since we don't have a 'clusters' table populated by an offline job yet,
    we will simulate clusters or return raw faces if clustering hasn't run.
    
    TODO: Implement the offline clustering job (DBSCAN/Chinese Whispers on embeddings).
    """
    # Placeholder: Return all faces as one "Unsorted" cluster for now
    # In reality, we'd query: SELECT * FROM faces WHERE cluster_id IS NOT NULL...
    
    # Let's just list the files in the faces dir for a quick UI test
    try:
        files = os.listdir(FACES_DIR)
        faces = [{"id": f, "url": f"/faces/{f}"} for f in files if f.endswith('.jpg')]
        return [{"id": 0, "name": "Unsorted", "faces": faces}]
    except Exception as e:
        return []

@app.post("/tag/cluster")
def tag_cluster(req: ClusterRequest):
    """Tags an entire cluster with a person's name."""
    # conn = get_db()
    # conn.execute("UPDATE faces SET person_name=? WHERE cluster_id=?", (req.name, req.cluster_id))
    # conn.commit()
    return {"status": "success", "message": f"Cluster {req.cluster_id} tagged as {req.name}"}

@app.get("/stats")
def get_stats():
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM media_files")
    total_files = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM media_files WHERE status='COMPLETED'")
    processed = cursor.fetchone()[0]
    
    return {
        "total_files": total_files,
        "processed": processed,
        "pending": total_files - processed
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
