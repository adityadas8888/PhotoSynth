#!/usr/bin/env python3
import numpy as np
from sklearn.cluster import DBSCAN
from photosynth.db import PhotoSynthDB

def main():
    print("🧠 Clustering Faces...")
    db = PhotoSynthDB()
    
    data = db.get_all_embeddings()
    if not data:
        print("No faces found.")
        return

    ids, embeddings = zip(*data)
    X = np.vstack(embeddings)
    
    print(f"📊 Analyzing {len(X)} faces...")
    
    # DBSCAN: epsilon=0.45 is a good sweet spot for ArcFace
    clt = DBSCAN(metric="cosine", eps=0.45, min_samples=3)
    clt.fit(X)
    
    labels = clt.labels_
    updates = [(int(cid), fid) for cid, fid in zip(labels, ids)]
    
    db.update_clusters(updates)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"✅ Created {n_clusters} clusters.")
    print(f"👉 Use SQLite/UI to name Cluster 0, Cluster 1, etc.")

if __name__ == "__main__":
    main()