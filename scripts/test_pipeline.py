#!/usr/bin/env python3
import os
import sys
import time
import threading
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.live import Live
from photosynth.tasks import run_detection_pass
from photosynth.db import PhotoSynthDB
from photosynth.utils.hashing import calculate_content_hash

# Config
TEST_DIR = Path(os.path.expanduser("~/personal/nas/video/TEST"))
EXTENSIONS = ['.jpg', '.jpeg', '.png', '.mp4', '.mov', '.mkv']

console = Console()

def should_skip(path):
    p = str(path)
    if '@eaDir' in p: return True
    if '/.' in p: return True
    if '#recycle' in p: return True
    return False

def get_file_status(file_hash):
    """Queries DB for current status of a file."""
    try:
        db = PhotoSynthDB()
        status = db.check_status(file_hash)
        return status if status else "QUEUED"
    except:
        return "UNKNOWN"

def generate_table(tasks):
    """Creates the rich table."""
    table = Table(title="🚀 PhotoSynth Pipeline Status")
    table.add_column("File", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Stage", style="green")

    # Icon map
    icons = {
        "QUEUED": "⏳",
        "PROCESSING_DETECTION": "👁️  (3090)",
        "PROCESSING_VLM": "🤖 (5090)",
        "COMPLETED": "✅",
        "ERROR_METADATA": "⚠️  Meta Err",
        "SKIPPED": "⏭️  Skip"
    }

    for task in tasks:
        status = get_file_status(task['hash'])
        # Map DB status to readable stage
        icon = icons.get(status, "❓")
        
        # Colorize
        style = "white"
        if status == "COMPLETED": style = "green"
        if "PROCESSING" in status: style = "bold yellow"
        
        table.add_row(
            task['name'], 
            status, 
            icon,
            style=style
        )
    return table

def main():
    if not TEST_DIR.exists():
        console.print(f"[red]❌ Test directory not found: {TEST_DIR}[/red]")
        sys.exit(1)

    # 1. Find Files
    files = []
    for ext in EXTENSIONS:
        files.extend(TEST_DIR.rglob(f"*{ext}"))
        files.extend(TEST_DIR.rglob(f"*{ext.upper()}"))
    files = [f for f in files if not should_skip(f)]
    files.sort()

    if not files:
        console.print("[yellow]⚠️  No files found.[/yellow]")
        sys.exit(0)

    # 2. Pre-calculate hashes and queue tasks (Two-Pass Batching)
    tasks = []
    console.print(f"[bold blue]📸 Found {len(files)} files. Queuing Batch 1: Detection...[/bold blue]")
    
    # Pass 1: Detection
    for f in files:
        f_hash = calculate_content_hash(str(f))
        if not f_hash: continue
        
        tasks.append({
            "name": f.name,
            "path": str(f),
            "hash": f_hash
        })
        
        # Dispatch Detection
        run_detection_pass.delay(str(f))

    console.print(f"[bold blue]🤖 Queuing Batch 2: Captioning...[/bold blue]")
    
    # Pass 2: Captioning
    # We reuse the calculated hashes/tasks list, but we need to dispatch caption tasks
    # Note: We dispatch them NOW. 
    # - In Distributed mode: 5090 picks them up immediately.
    # - In Single Node mode: Worker finishes Detections (FIFO), then picks these up.
    from photosynth.tasks import run_vlm_captioning
    
    for task in tasks:
        run_vlm_captioning.delay(task['path'])

    # 3. Live Monitor Loop
    with Live(generate_table(tasks), refresh_per_second=4) as live:
        while True:
            # Check if all are done
            all_done = True
            for task in tasks:
                status = get_file_status(task['hash'])
                if status not in ["COMPLETED", "SKIPPED", "ERROR_METADATA"]:
                    all_done = False
            
            # Update Table
            live.update(generate_table(tasks))
            
            if all_done:
                break
            time.sleep(0.5)

    console.print("\n[bold green]✨ All tasks finished![/bold green]")

if __name__ == "__main__":
    main()