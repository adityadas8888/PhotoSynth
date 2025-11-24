#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.live import Live
from photosynth.db import PhotoSynthDB
from photosynth.tasks import extract_faces_task
from photosynth.utils.hashing import calculate_content_hash

# Config
NAS_PATH = os.path.expanduser("~/personal/nas/homes/aditya")
EXTENSIONS = ['.jpg', '.jpeg', '.png', '.arw', '.heic']

console = Console()

def generate_table(tasks):
    table = Table(title="Face Scanning Progress")
    table.add_column("File", style="cyan")
    table.add_column("Hash", style="blue")
    table.add_column("Status", style="magenta")
    
    db = PhotoSynthDB()
    
    for t in tasks:
        # Check if faces exist for this hash
        conn = db.get_connection()
        with conn.cursor() as c:
            c.execute("SELECT COUNT(*) FROM faces WHERE file_hash=%s", (t['hash'],))
            face_count = c.fetchone()[0]
        conn.close()
        
        status = f"✅ {face_count} faces" if face_count > 0 else "⏳ Pending"
        table.add_row(t['name'], t['hash'][:16], status)
    
    return table

def main():
    console.print("[bold blue]🚀 Starting Distributed Face Harvest...[/bold blue]")
    db = PhotoSynthDB()
    
    # 1. Load Cache
    console.print("   Loading DB index...")
    conn = db.get_connection()
    
    with conn.cursor() as c:
        c.execute("SELECT file_path FROM media_files")
        known_paths = set(row[0] for row in c.fetchall())
        
        c.execute("SELECT DISTINCT file_hash FROM faces")
        known_hashes = set(row[0] for row in c.fetchall())
    
    conn.close()
    console.print(f"   Loaded {len(known_paths)} paths and {len(known_hashes)} hashes.")
    
    # 2. Find Files
    console.print("   Listing files on NAS...")
    files = []
    for ext in EXTENSIONS:
        files.extend(Path(NAS_PATH).rglob(f"*{ext}"))
        files.extend(Path(NAS_PATH).rglob(f"*{ext.upper()}"))
    
    console.print(f"[bold blue]📂 Found {len(files)} files. Queuing tasks...[/bold blue]")
    
    tasks = []
    path_skipped = 0
    hash_skipped = 0
    
    for p in files:
        path_str = str(p)
        if "@eaDir" in path_str: continue
        
        # Path check
        if path_str in known_paths:
            path_skipped += 1
            continue
        
        # Hash check
        f_hash = calculate_content_hash(path_str)
        if not f_hash: continue
        
        if f_hash in known_hashes:
            db.register_file(f_hash, path_str)
            hash_skipped += 1
            continue
        
        # Queue task
        tasks.append({
            "name": p.name,
            "path": path_str,
            "hash": f_hash
        })
        
        extract_faces_task.delay(path_str)
    
    console.print(f"[bold green]✅ Queued {len(tasks)} files for processing[/bold green]")
    console.print(f"⏩ Skipped (Path): {path_skipped}, Skipped (Hash): {hash_skipped}")
    
    if not tasks:
        console.print("[yellow]No new files to process.[/yellow]")
        return
    
    # 3. Monitor Progress
    console.print("\n[bold blue]📊 Monitoring progress (Ctrl+C to exit)...[/bold blue]\n")
    
    with Live(generate_table(tasks), refresh_per_second=2) as live:
        try:
            while True:
                live.update(generate_table(tasks))
                time.sleep(2)
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitoring stopped. Tasks continue in background.[/yellow]")
    
    console.print("\n[bold green]👉 Run: uv run python scripts/cluster_faces.py[/bold green]")

if __name__ == "__main__":
    main()