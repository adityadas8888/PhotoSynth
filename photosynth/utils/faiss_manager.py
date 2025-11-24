import faiss
import numpy as np
import os
from pathlib import Path
from photosynth.db import PhotoSynthDB
import time

# --- CONFIGURATION ---
# Store the index file in a hidden directory in your home folder
INDEX_DIR = Path(os.path.expanduser("~/.photosynth/"))
INDEX_FILE = INDEX_DIR / "face_index.faiss"
ID_MAP_FILE = INDEX_DIR / "face_id_map.npy"
# Similarity score (0.0 to 1.0) required to consider a face 'known'. Adjust this based on accuracy tests.
SIMILARITY_THRESHOLD = 0.7


# ---------------------

class FAISSManager:
    def __init__(self):
        self.index = None
        self.face_id_map = None  # Maps FAISS internal index position to DB face_id
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        self._load_index()  # Attempt to load on initialization

    def _load_index(self):
        """Loads the index and ID map from disk."""
        if INDEX_FILE.exists() and ID_MAP_FILE.exists():
            try:
                self.index = faiss.read_index(str(INDEX_FILE))
                self.face_id_map = np.load(ID_MAP_FILE)

                # Move index to GPU immediately upon loading (for the 5090 worker)
                if faiss.get_num_gpus() > 0:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                    print(f"FAISS index loaded and moved to GPU 0 (5090/3090).")

                print(f"Loaded FAISS index with {self.index.ntotal} vectors.")
                return True
            except Exception as e:
                # If loading fails (e.g., corruption, wrong version), trigger a rebuild
                print(f"Error loading FAISS index: {e}. Rebuilding from scratch...")
                self.index = None
                return False
        return False

    def build_index_if_missing(self):
        """Builds index from DB if index file doesn't exist or load fails."""
        if self.index:
            return

        print("Starting FAISS index rebuild from PostgreSQL...")
        db = PhotoSynthDB()
        face_data = db.get_all_embeddings()  # Fetches [(face_id, embedding), ...]

        if not face_data:
            print("No faces found in DB. Index not built.")
            return

        self.face_id_map = np.array([d[0] for d in face_data], dtype=np.int64)
        embeddings = np.array([d[1] for d in face_data], dtype=np.float32)

        # IndexFlatIP is simple and effective for inner product similarity (cosine-like)
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)

        # Add vectors and move to GPU
        index.add(embeddings)
        if faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print(f"FAISS index built and moved to GPU 0.")

        self.index = index
        self._save_index()
        print(f"FAISS Index successfully built with {index.ntotal} vectors.")

    def _save_index(self):
        """Saves the index and ID map to disk."""
        if self.index:
            # Move index back to CPU before saving if it was on GPU
            index_to_save = self.index
            if faiss.get_num_gpus() > 0:
                index_to_save = faiss.index_gpu_to_cpu(self.index)

            faiss.write_index(index_to_save, str(INDEX_FILE))
            np.save(ID_MAP_FILE, self.face_id_map)
            print(f"FAISS Index saved to disk. Total faces: {self.index.ntotal}")

    def search_face(self, query_embedding, k=1):
        """
        Searches the index for the nearest neighbor.
        Returns (matched_face_id, cluster_id) if similarity > threshold.
        """
        if self.index is None:
            self.build_index_if_missing()
            if self.index is None: return None, None

        # Reshape query for FAISS (needs a 2D array)
        query = query_embedding.astype(np.float32).reshape(1, -1)

        # D = Similarity Score, I = Index Position
        D, I = self.index.search(query, k)

        similarity_score = D[0][0]
        faiss_index = I[0][0]

        # Check for a valid match and required similarity
        if faiss_index != -1 and similarity_score >= SIMILARITY_THRESHOLD:
            # Map the FAISS index position back to the PostgreSQL face_id
            matched_face_id = self.face_id_map[faiss_index]

            # Retrieve the cluster_id from the database based on the matched face_id
            db = PhotoSynthDB()
            conn = db.get_connection()
            with conn.cursor() as c:
                c.execute("SELECT cluster_id FROM faces WHERE face_id=%s", (matched_face_id,))
                cluster_id = c.fetchone()[0]
            conn.close()

            return matched_face_id, cluster_id

        return None, None


# Singleton pattern for the worker to load the index only once
faiss_manager_instance = None


def get_faiss_manager():
    global faiss_manager_instance
    if faiss_manager_instance is None:
        faiss_manager_instance = FAISSManager()
    return faiss_manager_instance