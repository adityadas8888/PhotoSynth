#!/usr/bin/env python3
import time
import numpy as np
import faiss
from rich.console import Console
from photosynth.db import PhotoSynthDB
from photosynth.tasks import run_clustering_task

# Configuration
console = Console()


def main():
    console.print("[bold blue]🧠 Starting Face Clustering Pipeline...[/bold blue]")
    db = PhotoSynthDB()

    # 1. Fetch ALL faces count (just to determine if work is needed)
    console.print("   Checking total face embeddings in DB...")

    # Your db.get_all_embeddings returns a list, so we calculate the count here
    all_face_data = db.get_all_embeddings()

    if not all_face_data:
        console.print("[yellow]No faces found to cluster. Run scan_faces.py first.[/yellow]")
        return

    total_embeddings_count = len(all_face_data)

    console.print(f"   Found {total_embeddings_count} total embeddings.")

    # 2. Queue the heavy lifting task to the dedicated worker
    console.print(f"[bold magenta]📦 Queuing GPU clustering task to 5090 worker...[/bold magenta]")

    # We pass the count to the task, which will re-fetch the data for safety/scalability.
    task_result = run_clustering_task.apply_async(
        args=[total_embeddings_count],
        queue='face_queue'  # Targets the 5090
    )

    console.print(f"✅ Clustering task started: ID {task_result.id}")

    # 3. Monitor (simple monitor for the clustering task)
    console.print("\n[bold blue]📊 Monitoring Clustering Progress (Ctrl+C to exit)...[/bold blue]\n")

    try:
        while not task_result.ready():
            status = task_result.status
            console.print(f"Current Status: [yellow]{status}[/yellow]...", end='\r')
            time.sleep(5)

        final_status = task_result.get()
        console.print(f"\n[bold green]🏁 Clustering Complete![/bold green] {final_status}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped. Clustering continues in background.[/yellow]")

    except Exception as e:
        console.print(f"\n[red]❌ Clustering failed: {e}[/red]")


if __name__ == "__main__":
    main()