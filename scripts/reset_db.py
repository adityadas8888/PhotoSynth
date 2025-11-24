def reset_db():
    print(f"🧨 RESETTING POSTGRES DATABASE...")
    
    from photosynth.db import PhotoSynthDB
    db = PhotoSynthDB()
    conn = db.get_connection()
    
    tables = ['faces', 'people', 'media_files']
    
    try:
        with conn.cursor() as c:
            for t in tables:
                print(f"   🗑️ Dropping table {t}...")
                c.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
        conn.commit()
        print("✨ Tables dropped.")
        
        # Re-init
        print("✨ Re-initializing schema...")
        db._init_db()
        print("✅ Database reset complete!")
        
    except Exception as e:
        print(f"❌ Reset failed: {e}")
    finally:
        conn.close()
