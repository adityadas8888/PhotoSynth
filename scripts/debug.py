from photosynth.db import PhotoSynthDB

db = PhotoSynthDB()
conn = db.get_connection()
cursor = conn.cursor()

# 1. Count total files processed
cursor.execute("SELECT COUNT(*) FROM media_files")
file_count = cursor.fetchone()[0]

# 2. Count total faces
cursor.execute("SELECT COUNT(*) FROM faces")
face_count = cursor.fetchone()[0]

print(f"📊 DB Report: {file_count} Files Processed | {face_count} Faces Found")

# 3. Show me the files!
print("\n📂 First 20 files in the database:")
cursor.execute("SELECT file_path FROM media_files LIMIT 20")
for row in cursor.fetchall():
    print(f" - {row[0]}")

conn.close()