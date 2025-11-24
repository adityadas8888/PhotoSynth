import psycopg2
import psycopg2.extras
import time
import os
import numpy as np
import io
import json

# --- CONFIGURATION ---
DB_HOST = "10.0.0.230"
DB_NAME = "photosynth"
DB_USER = "photosynth"
DB_PASS = "secure_password_123"

class PhotoSynthDB:
    def __init__(self):
        self._init_db()

    def get_connection(self):
        return psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )

    def _init_db(self):
        conn = self.get_connection()
        c = conn.cursor()
        
        # Media Files Table
        c.execute('''
            CREATE TABLE IF NOT EXISTS media_files (
                file_hash TEXT PRIMARY KEY,
                file_path TEXT,
                status TEXT DEFAULT 'PENDING',
                
                -- Granular Status Tracking
                detection_status TEXT DEFAULT 'PENDING',
                caption_status TEXT DEFAULT 'PENDING',
                
                -- Intermediate Results (JSON)
                detection_data JSONB,
                caption_data JSONB,
                
                vlm_narrative TEXT,
                search_concepts JSONB,
                last_updated REAL
            )
        ''')

        # People Table
        c.execute('''
            CREATE TABLE IF NOT EXISTS people (
                cluster_id INTEGER PRIMARY KEY,
                name TEXT DEFAULT 'Unknown'
            )
        ''')

        # Faces Table
        c.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                face_id SERIAL PRIMARY KEY,
                file_hash TEXT,
                embedding BYTEA,
                cluster_id INTEGER DEFAULT -1,
                FOREIGN KEY(file_hash) REFERENCES media_files(file_hash)
            )
        ''')
        conn.commit()
        conn.close()

    def register_file(self, file_hash, file_path):
        from photosynth.utils.paths import make_relative
        rel_path = make_relative(file_path)
        
        conn = self.get_connection()
        try:
            with conn.cursor() as c:
                c.execute('''
                    INSERT INTO media_files (file_hash, file_path, status, last_updated)
                    VALUES (%s, %s, 'PENDING', %s)
                    ON CONFLICT (file_hash) DO UPDATE SET file_path=EXCLUDED.file_path
                ''', (file_hash, rel_path, time.time()))
            conn.commit()
        except Exception as e:
            print(f"DB Error: {e}")
        finally:
            conn.close()

    def update_status(self, file_hash, status, narrative=None, concepts=None):
        conn = self.get_connection()
        updates = ["status=%s", "last_updated=%s"]
        params = [status, time.time()]
        
        if narrative:
            updates.append("vlm_narrative=%s")
            params.append(narrative)
        if concepts:
            updates.append("search_concepts=%s")
            params.append(json.dumps(concepts))
            
        params.append(file_hash)
        
        with conn.cursor() as c:
            c.execute(f"UPDATE media_files SET {', '.join(updates)} WHERE file_hash=%s", params)
        conn.commit()
        conn.close()

    def update_detection_result(self, file_hash, status, data=None):
        conn = self.get_connection()
        json_data = json.dumps(data) if data else None
        with conn.cursor() as c:
            c.execute('''
                UPDATE media_files 
                SET detection_status=%s, detection_data=%s, last_updated=%s 
                WHERE file_hash=%s
            ''', (status, json_data, time.time(), file_hash))
        conn.commit()
        conn.close()

    def update_caption_result(self, file_hash, status, data=None):
        conn = self.get_connection()
        json_data = json.dumps(data) if data else None
        with conn.cursor() as c:
            c.execute('''
                UPDATE media_files 
                SET caption_status=%s, caption_data=%s, last_updated=%s 
                WHERE file_hash=%s
            ''', (status, json_data, time.time(), file_hash))
        conn.commit()
        conn.close()

    def get_file_data(self, file_hash):
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as c:
                c.execute("SELECT * FROM media_files WHERE file_hash=%s", (file_hash,))
                row = c.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def check_status(self, file_hash):
        conn = self.get_connection()
        try:
            with conn.cursor() as c:
                c.execute("SELECT status FROM media_files WHERE file_hash=%s", (file_hash,))
                row = c.fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def add_face(self, file_hash, embedding):
        # Convert numpy to bytes
        emb_bytes = embedding.tobytes()
        conn = self.get_connection()
        with conn.cursor() as c:
            c.execute('INSERT INTO faces (file_hash, embedding) VALUES (%s, %s)', (file_hash, emb_bytes))
        conn.commit()
        conn.close()

    def get_all_embeddings(self):
        conn = self.get_connection()
        with conn.cursor() as c:
            c.execute('SELECT face_id, embedding FROM faces')
            rows = c.fetchall()
        conn.close()
        # Convert bytes back to numpy
        return [(r[0], np.frombuffer(r[1], dtype=np.float32)) for r in rows]

    def update_clusters(self, cluster_map):
        conn = self.get_connection()
        with conn.cursor() as c:
            # Batch update
            psycopg2.extras.execute_batch(c, 
                'UPDATE faces SET cluster_id = %s WHERE face_id = %s',
                cluster_map
            )
            
            unique = set(c_id for c_id, f_id in cluster_map if c_id != -1)
            for c_id in unique:
                c.execute('INSERT INTO people (cluster_id) VALUES (%s) ON CONFLICT (cluster_id) DO NOTHING', (c_id,))
        conn.commit()
        conn.close()

    def get_known_faces(self):
        conn = self.get_connection()
        with conn.cursor() as c:
            c.execute('''
                SELECT f.cluster_id, p.name, f.embedding 
                FROM faces f
                JOIN people p ON f.cluster_id = p.cluster_id
                WHERE f.cluster_id != -1
            ''')
            rows = c.fetchall()
        conn.close()
        return [(r[0], r[1], np.frombuffer(r[2], dtype=np.float32)) for r in rows]