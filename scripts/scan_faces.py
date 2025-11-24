#!/usr/bin/env python3
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
HASH_WORKERS = 16
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
    console.print("   Loading DB index...")
    conn = db.get_connection()

    with conn.cursor() as c:
        c.execute("SELECT file_path FROM media_files")
        known_paths = set(row[0] for row in c.fetchall())

        c.execute("SELECT DISTINCT file_hash FROM faces")
        known_hashes = set(row[0] for row in c.fetchall())

    conn.close()
    console.print(f"   Loaded {len(known_paths)} paths and {len(known_hashes)} hashes.")

    # 2. Find Files
    console.print("   Listing files on NAS...")
    files = []
    for ext in EXTENSIONS:
        files.extend(Path(NAS_PATH).rglob(f"*{ext}"))
        files.extend(Path(NAS_PATH).rglob(f"*{ext.upper()}"))

    console.print(f"[bold blue]📂 Found {len(files)} files. Starting parallel hash calculation...[/bold blue]")

    tasks = []
    path_skipped = 0
    hash_skipped = 0

    # --- START OF PARALLEL HASHING AND BATCH QUEUEING LOGIC ---
    files_to_register = []
    files_to_process = []

    with ThreadPoolExecutor(max_workers=HASH_WORKERS) as executor:
        # Submit all hash calculations to the thread pool
        future_to_path = {
            executor.submit(calculate_content_hash, str(p)): str(p)
            for p in files if "@eaDir" not in str(p)
        }

        # Process results as they complete (faster than waiting for all)
        for future in as_completed(future_to_path):
            path_str = future_to_path[future]

            try:
                f_hash = future.result()
            except Exception as exc:
                # Handle potential hashing failures (e.g., corrupted files)
                console.print(f"[red]⚠️ Hashing failed for {path_str}: {exc}[/red]")
                f_hash = None

            if not f_hash: continue

            # Path check
            if path_str in known_paths:
                path_skipped += 1
                continue

            # Hash check: if hash is known, we register it (to update the path)
            # but skip the processing task, then move on.
            if f_hash in known_hashes:
                files_to_register.append((f_hash, path_str))
                hash_skipped += 1
                continue

            # Queue task for processing and register file
            files_to_register.append((f_hash, path_str))
            files_to_process.append(path_str)  # Path for Celery delay

            # Prepare task entry for the monitoring table
            tasks.append({
                "name": Path(path_str).name,
                "path": path_str,
                "hash": f_hash
            })

    # --- BATCH DATABASE REGISTRATION (Sequential but highly efficient) ---
    if files_to_register:
        console.print(f"💾 Batch registering/updating {len(files_to_register)} file paths...")
        db.batch_register_files(files_to_register)

    # --- BATCH TASK QUEUEING (Still using Celery's .delay/.apply_async) ---
    for path_str in files_to_process:
        # Use apply_async to specify a dedicated queue for the 5090 worker
        # Assuming you've configured a 'face_queue' (see section 2 of the previous answer)
        extract_faces_task.apply_async(args=[path_str], queue='face_queue')

    console.print(f"[bold green]✅ Queued {len(files_to_process)} files for processing[/bold green]")
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