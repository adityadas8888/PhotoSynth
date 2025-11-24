import os
import sqlite3

DB_PATH = os.path.expanduser("~/personal/nas/homes/photosynth.db")

def reset_db():
    print(f"🧨 RESETTING DATABASE at {DB_PATH}...")
    
    # 1. Delete Files
    files_to_delete = [DB_PATH, DB_PATH + "-wal", DB_PATH + "-shm"]
    for f in files_to_delete:
        if os.path.exists(f):
            try:
                os.remove(f)
                print(f"   🗑️ Deleted {f}")
            except Exception as e:
                print(f"   ❌ Failed to delete {f}: {e}")
    
    # 2. Re-initialize DB (using PhotoSynthDB class)
    print("✨ Re-initializing empty database...")
    from photosynth.db import PhotoSynthDB
    db = PhotoSynthDB()
    
    # 3. Verify Tables
    conn = db.get_connection()
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    conn.close()
    
    print(f"✅ Database reset complete! Tables: {[t[0] for t in tables]}")

if __name__ == "__main__":
    reset_db()
