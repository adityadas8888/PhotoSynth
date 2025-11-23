import sqlite3
import time
import os
import numpy as np
import io

DB_PATH = os.path.expanduser("~/personal/PhotoSynth/photosynth.db")

# --- Numpy Adapters for SQLite ---
# Allows us to save face embeddings (lists of floats) directly into the DB
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
        conn = self.get_connection()
        c = conn.cursor()
        
        # 1. Media Files (The Master List)
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

        # 2. People / Clusters (Who is this?)
        # cluster_id 0 might be 'Aditya', 1 might be 'Ankita'
        c.execute('''
            CREATE TABLE IF NOT EXISTS people (
                cluster_id INTEGER PRIMARY KEY,
                name TEXT DEFAULT 'Unknown'
            )
        ''')

        # 3. Faces (The Raw Data)
        # Stores the 512-dim vector for every face found in every file
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
        # detect_types is needed to trigger the numpy converter
        return sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)

    # --- File Operations ---
    def register_file(self, file_hash, file_path):
        conn = self.get_connection()
        try:
            conn.execute('''
                INSERT INTO media_files (file_hash, file_path, status, last_updated)
                VALUES (?, ?, 'PENDING', ?)
                ON CONFLICT(file_hash) DO UPDATE SET file_path=excluded.file_path
            ''', (file_hash, file_path, time.time()))
            conn.commit()
        except Exception as e:
            print(f"DB Error: {e}")
        finally:
            conn.close()

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
        sql = f"UPDATE media_files SET {', '.join(updates)} WHERE file_hash=?"
        
        conn.execute(sql, params)
        conn.commit()
        conn.close()

    def check_status(self, file_hash):
        conn = self.get_connection()
        c = conn.execute("SELECT status FROM media_files WHERE file_hash=?", (file_hash,))
        row = c.fetchone()
        conn.close()
        return row[0] if row else None

    # --- Face Operations ---
    def add_face(self, file_hash, embedding):
        """Saves a raw face embedding."""
        conn = self.get_connection()
        conn.execute('INSERT INTO faces (file_hash, embedding) VALUES (?, ?)', (file_hash, embedding))
        conn.commit()
        conn.close()

    def get_all_embeddings(self):
        """Used by the Clustering Script."""
        conn = self.get_connection()
        cursor = conn.execute('SELECT face_id, embedding FROM faces')
        data = cursor.fetchall()
        conn.close()
        return data

    def update_clusters(self, cluster_map):
        """
        Updates faces with their calculated Cluster ID.
        cluster_map: list of (cluster_id, face_id)
        """
        conn = self.get_connection()
        # Bulk update faces
        conn.executemany('UPDATE faces SET cluster_id = ? WHERE face_id = ?', cluster_map)
        
        # Register new people in the 'people' table
        unique_clusters = set(c_id for c_id, _ in cluster_map if c_id != -1)
        for c_id in unique_clusters:
            conn.execute('INSERT OR IGNORE INTO people (cluster_id) VALUES (?)', (c_id,))
            
        conn.commit()
        conn.close()

    def get_known_faces(self):
        """
        Used by the Detector.
        Returns [(cluster_id, name, embedding), ...] for all identified people.
        """
        conn = self.get_connection()
        # We join faces and people to get names
        query = '''
            SELECT f.cluster_id, p.name, f.embedding 
            FROM faces f
            JOIN people p ON f.cluster_id = p.cluster_id
            WHERE f.cluster_id != -1
        '''
        rows = conn.execute(query).fetchall()
        conn.close()
        return rows