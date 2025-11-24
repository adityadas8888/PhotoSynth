import sqlite3
import os

# Shared NAS Path
DB_PATH = os.path.expanduser("~/personal/nas/homes/photosynth.db")

def migrate():
    print(f"📂 Migrating database at {DB_PATH}...")
    
    if not os.path.exists(DB_PATH):
        print("❌ Database not found!")
        return

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    columns = [
        ("detection_status", "TEXT DEFAULT 'PENDING'"),
        ("caption_status", "TEXT DEFAULT 'PENDING'"),
        ("detection_data", "TEXT"),
        ("caption_data", "TEXT")
    ]
    
    for col_name, col_def in columns:
        try:
            print(f"   ➕ Adding column: {col_name}...")
            c.execute(f"ALTER TABLE media_files ADD COLUMN {col_name} {col_def}")
        except sqlite3.OperationalError as e:
            if "duplicate column" in str(e):
                print(f"      ⚠️ Column {col_name} already exists.")
            else:
                print(f"      ❌ Error adding {col_name}: {e}")

    conn.commit()
    conn.close()
    print("✅ Migration Complete!")

if __name__ == "__main__":
    migrate()
