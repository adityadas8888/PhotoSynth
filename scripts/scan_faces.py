#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.live import Live
from concurrent.futures import ThreadPoolExecutor, as_completed
from celery.result import AsyncResult  # Added for monitoring
from photosynth.db import PhotoSynthDB
from photosynth.tasks import extract_faces_task
from photosynth.utils.hashing import calculate_content_hash

# Config
NAS_PATH = os.path.expanduser("~/personal/nas/homes/aditya")
EXTENSIONS = ['.jpg', '.jpeg', '.png', '.arw', '.heic']
HASH_WORKERS = 16  # Configured for fast, parallel I/O and hashing

console = Console()


def generate_table(tasks):
    table = Table(title="Face Scanning Progress")
    table.add_column("File", style="cyan")
    table.add_column("Hash", style="blue")
    table.add_column("DB Faces", style="magenta")
    table.add_column("Celery Status", style="yellow")

    db = PhotoSynthDB()

    for t in tasks:
        # Check if faces exist for this hash
        conn = db.get_connection()
        with conn.cursor() as c:
            c.execute("SELECT COUNT(*) FROM faces WHERE file_hash=%s", (t['hash'],))
            face_count = c.fetchone()[0]
        conn.close()

        db_status = f"✅ {face_count} faces" if face_count > 0 else "⏳ Pending DB"

        # Celery Status Check
        celery_status = "N/A"
        if t.get('task_id'):
            result = AsyncResult(t['task_id'])
            celery_status = str(result.status)
            if result.status == 'SUCCESS':
                celery_status = f"[green]{result.status}[/green]"
            elif result.status == 'STARTED':
                celery_status = f"[yellow]{result.status}[/yellow]"
            elif result.status in ('PENDING', 'RECEIVED'):
                celery_status = f"[cyan]{result.status}[/cyan]"
            elif result.status in ('FAILURE', 'REVOKED'):
                celery_status = f"[red]{result.status}[/red]"

        table.add_row(t['name'], t['hash'][:16], db_status, celery_status)

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

    # 2. Find Files (Using the debug-friendly loop for better visibility)
    console.print("   Listing files on NAS...")
    files = []
    file_count = 0
    start_time = time.time()

    for ext in EXTENSIONS:
        # Search for lowercase extensions
        for p in Path(NAS_PATH).rglob(f"*{ext}"):
            files.append(p)
            file_count += 1

        # Search for uppercase extensions
        ext_upper = ext.upper()
        for p in Path(NAS_PATH).rglob(f"*{ext_upper}"):
            files.append(p)
            file_count += 1

    console.print(f"[bold blue]📂 Found {len(files)} files. Starting parallel hash calculation...[/bold blue]")

    tasks_for_monitor = []
    files_to_register = []
    files_to_queue = []

    path_skipped = 0
    hash_skipped = 0

    # --- PARALLEL HASHING AND FILTERING ---
    with ThreadPoolExecutor(max_workers=HASH_WORKERS) as executor:
        # Submit all hash calculations to the thread pool, filtering out the @eaDir paths
        future_to_path = {
            executor.submit(calculate_content_hash, str(p)): str(p)
            for p in files if "@eaDir" not in str(p)
        }

        for future in as_completed(future_to_path):
            path_str = future_to_path[future]

            try:
                f_hash = future.result()
            except Exception as exc:
                console.print(f"[red]⚠️ Hashing failed for {path_str}: {exc}[/red]")
                f_hash = None

            if not f_hash: continue

            # Path check
            if path_str in known_paths:
                path_skipped += 1
                continue

            # Hash check: if hash is known, register the file (to update path) but skip the task
            if f_hash in known_hashes:
                files_to_register.append((f_hash, path_str))
                hash_skipped += 1
                continue

            # New file/hash: Register and Queue
            files_to_register.append((f_hash, path_str))
            files_to_queue.append((f_hash, path_str))

    # --- BATCH DATABASE REGISTRATION ---
    if files_to_register:
        console.print(f"💾 Batch registering/updating {len(files_to_register)} file paths...")
        db.batch_register_files(files_to_register)

    # --- TASK QUEUEING AND MONITOR SETUP ---
    for f_hash, path_str in files_to_queue:
        # Queue task with specific routing for the 5090 (face_queue)
        task_result = extract_faces_task.apply_async(args=[path_str], queue='face_queue')

        # Prepare task entry for the monitoring table
        tasks_for_monitor.append({
            "name": Path(path_str).name,
            "path": path_str,
            "hash": f_hash,
            "task_id": task_result.id
        })

    console.print(f"[bold green]✅ Queued {len(files_to_queue)} files for processing[/bold green]")
    console.print(f"⏩ Skipped (Path): {path_skipped}, Skipped (Hash): {hash_skipped}")

    if not tasks_for_monitor:
        console.print("[yellow]No new files to process.[/yellow]")
        return

    # 3. Monitor Progress
    console.print("\n[bold blue]📊 Monitoring progress (Ctrl+C to exit)...[/bold blue]\n")

    with Live(generate_table(tasks_for_monitor), refresh_per_second=2) as live:
        try:
            while True:
                live.update(generate_table(tasks_for_monitor))
                time.sleep(2)
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitoring stopped. Tasks continue in background.[/yellow]")

    console.print("\n[bold green]👉 Run: uv run python scripts/cluster_faces.py[/bold green]")


if __name__ == "__main__":
    main()