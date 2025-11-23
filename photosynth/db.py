import sqlite3
import time
import os
import numpy as np
import io

# --- 1. SHARED NAS PATH ---
DB_PATH = os.path.expanduser("~/personal/nas/photosynth.db")

# --- Numpy Adapters ---
def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return out.read()

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("ARRAY", convert_array)

class PhotoSynthDB:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        # Ensure NAS folder exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = self.get_connection()
        c = conn.cursor()
        
        # --- 2. WAL MODE (Critical for NAS concurrency) ---
        c.execute("PRAGMA journal_mode=WAL;")
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS media_files (
                file_hash TEXT PRIMARY KEY,
                file_path TEXT,
                status TEXT DEFAULT 'PENDING',
                vlm_narrative TEXT,
                search_concepts TEXT,
                last_updated REAL
            )
        ''')

        c.execute('''
            CREATE TABLE IF NOT EXISTS people (
                cluster_id INTEGER PRIMARY KEY,
                name TEXT DEFAULT 'Unknown'
            )
        ''')

        c.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                face_id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash TEXT,
                embedding ARRAY,
                cluster_id INTEGER DEFAULT -1,
                FOREIGN KEY(file_hash) REFERENCES media_files(file_hash)
            )
        ''')
        conn.commit()
        conn.close()

    def get_connection(self):
        # --- 3. TIMEOUT (Prevents locks on slow network) ---
        return sqlite3.connect(self.db_path, timeout=30.0, detect_types=sqlite3.PARSE_DECLTYPES)

    def register_file(self, file_hash, file_path):
        conn = self.get_connection()
        try:
            conn.execute('''
                INSERT INTO media_files (file_hash, file_path, status, last_updated)
                VALUES (?, ?, 'PENDING', ?)
                ON CONFLICT(file_hash) DO UPDATE SET file_path=excluded.file_path
            ''', (file_hash, file_path, time.time()))
            conn.commit()
        except: pass
        finally: conn.close()

    def update_status(self, file_hash, status, narrative=None, concepts=None):
        conn = self.get_connection()
        updates = ["status=?, last_updated=?"]
        params = [status, time.time()]
        
        if narrative:
            updates.append("vlm_narrative=?")
            params.append(narrative)
        if concepts:
            updates.append("search_concepts=?")
            import json
            params.append(json.dumps(concepts))
            
        params.append(file_hash)
        conn.execute(f"UPDATE media_files SET {', '.join(updates)} WHERE file_hash=?", params)
        conn.commit()
        conn.close()

    def check_status(self, file_hash):
        conn = self.get_connection()
        row = conn.execute("SELECT status FROM media_files WHERE file_hash=?", (file_hash,)).fetchone()
        conn.close()
        return row[0] if row else None

    def add_face(self, file_hash, embedding):
        conn = self.get_connection()
        conn.execute('INSERT INTO faces (file_hash, embedding) VALUES (?, ?)', (file_hash, embedding))
        conn.commit()
        conn.close()

    def get_all_embeddings(self):
        conn = self.get_connection()
        data = conn.execute('SELECT face_id, embedding FROM faces').fetchall()
        conn.close()
        return data

    def update_clusters(self, cluster_map):
        conn = self.get_connection()
        conn.executemany('UPDATE faces SET cluster_id = ? WHERE face_id = ?', cluster_map)
        unique = set(c for c, f in cluster_map if c != -1)
        for c_id in unique:
            conn.execute('INSERT OR IGNORE INTO people (cluster_id) VALUES (?)', (c_id,))
        conn.commit()
        conn.close()

    def get_known_faces(self):
        conn = self.get_connection()
        rows = conn.execute('''
            SELECT f.cluster_id, p.name, f.embedding 
            FROM faces f
            JOIN people p ON f.cluster_id = p.cluster_id
            WHERE f.cluster_id != -1
        ''').fetchall()
        conn.close()
        return rows