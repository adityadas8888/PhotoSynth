import sqlite3
import os
import shutil

DB_PATH = os.path.expanduser("~/personal/nas/homes/photosynth.db")
BACKUP_PATH = DB_PATH + ".corrupt.bak"

def repair():
    print(f"🚑 Attempting to repair database at {DB_PATH}...")
    
    if not os.path.exists(DB_PATH):
        print("❌ Database not found!")
        return

    # 1. Backup corrupted DB
    print(f"📦 Backing up to {BACKUP_PATH}...")
    shutil.copy2(DB_PATH, BACKUP_PATH)
    
    try:
        # 2. Dump data
        print("📥 Dumping data...")
        conn = sqlite3.connect(DB_PATH)
        dump_sql = "\n".join(conn.iterdump())
        conn.close()
        
        # 3. Recreate DB
        print("♻️ Recreating database...")
        os.remove(DB_PATH)
        
        # Remove WAL files if they exist
        if os.path.exists(DB_PATH + "-wal"): os.remove(DB_PATH + "-wal")
        if os.path.exists(DB_PATH + "-shm"): os.remove(DB_PATH + "-shm")
        
        conn_new = sqlite3.connect(DB_PATH)
        conn_new.executescript(dump_sql)
        conn_new.close()
        
        print("✅ Database repaired successfully!")
        
    except Exception as e:
        print(f"❌ Repair failed: {e}")
        print("⚠️ Restoring backup...")
        shutil.copy2(BACKUP_PATH, DB_PATH)

if __name__ == "__main__":
    repair()
