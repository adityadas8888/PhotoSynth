import sqlite3
import json
import time
import os
from pathlib import Path

DB_PATH = os.path.expanduser("~/personal/PhotoSynth/photosynth.db")

class PhotoSynthDB:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS media_files (
                file_hash TEXT PRIMARY KEY,
                file_path TEXT,
                status TEXT DEFAULT 'PENDING',
                vlm_narrative TEXT,
                search_concepts TEXT,
                face_data TEXT,
                last_updated REAL
            )
        ''')
        conn.commit()
        conn.close()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def register_file(self, file_hash, file_path):
        """Registers a new file or updates its path if it exists."""
        conn = self.get_connection()
        c = conn.cursor()
        try:
            c.execute('''
                INSERT INTO media_files (file_hash, file_path, status, last_updated)
                VALUES (?, ?, 'PENDING', ?)
                ON CONFLICT(file_hash) DO UPDATE SET
                    file_path=excluded.file_path,
                    last_updated=excluded.last_updated
            ''', (file_hash, file_path, time.time()))
            conn.commit()
        except Exception as e:
            print(f"DB Error: {e}")
        finally:
            conn.close()

    def update_status(self, file_hash, status, narrative=None, concepts=None):
        """Updates the processing status and results."""
        conn = self.get_connection()
        c = conn.cursor()
        
        updates = ["status=?, last_updated=?"]
        params = [status, time.time()]
        
        if narrative:
            updates.append("vlm_narrative=?")
            params.append(narrative)
        
        if concepts:
            updates.append("search_concepts=?")
            params.append(json.dumps(concepts))
            
        params.append(file_hash)
        
        sql = f"UPDATE media_files SET {', '.join(updates)} WHERE file_hash=?"
        
        try:
            c.execute(sql, params)
            conn.commit()
        except Exception as e:
            print(f"DB Error: {e}")
        finally:
            conn.close()

    def check_status(self, file_hash):
        """Returns the status of a file."""
        conn = self.get_connection()
        c = conn.cursor()
        c.execute("SELECT status FROM media_files WHERE file_hash=?", (file_hash,))
        row = c.fetchone()
        conn.close()
        return row[0] if row else None
