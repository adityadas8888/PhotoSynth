# 📸 PhotoSynth: Advanced NAS Photo Tagging Pipeline

PhotoSynth is a distributed media processing pipeline designed to automatically enrich photo and video files stored on a Synology NAS with deep, searchable metadata. It leverages state-of-the-art Vision-Language Models (VLM) for advanced captioning and object detection, far surpassing standard indexing capabilities.

## 🚀 Architecture

The pipeline uses a Producer/Consumer model across two specialized machines to handle high-demand ML tasks efficiently:

| Component | Function | Running On |
| :--- | :--- | :--- |
| **Ingestion** | File watcher detects new media and pushes job to queue. | NAS/Lightweight Service |
| **Detection Pass** | Runs Object Detection (Grounding DINO + SAM 3), OCR, and Face Clustering. | **RTX 3090 PC** |
| **Captioning Pass** | Runs VLM (BLIP-2/LLaVA/vLLM) for detailed context and descriptions. | **RTX 5090 PC** |
| **Metadata Commitment** | Writes final metadata (tags, regions, captions) to file EXIF/XMP headers via ExifTool. | **5090 PC** (Final Stage) |

## ⚙️ Requirements

### Hardware / Infrastructure

* **Synology NAS:** Running DSM 7.x, configured with **NFS V4.1** and Read/Write access permissions.
* **Processing Node 1 (3090 PC):** Linux OS (recommended: Fedora/Ubuntu), **24GB+ VRAM** for detection models.
* **Processing Node 2 (5090 PC):** Linux OS (recommended: Fedora/Ubuntu), **32GB+ VRAM** for VLM models.
* **Network:** Both PCs must have NFS mounts to the media shares:
    * `~/personal/nas/photo`
    * `~/personal/nas/video`
    * `~/personal/nas/homes`
* **Queue:** A distributed queueing system (e.g., Redis or RabbitMQ) accessible by both processing nodes.

### Software

* Python 3.10+
* PyTorch (CUDA enabled)
* ExifTool (must be installed on the commit machine)
* Required Python libraries (see `requirements.txt`)

## 🛠️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/AdityaDas/PhotoSynth.git](https://github.com/AdityaDas/PhotoSynth.git)
    cd PhotoSynth
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure Environment:**
    * Copy `config/settings.yaml.example` to `config/settings.yaml`.
    * Update the file with your NAS IP and file paths.
    * Configure your queue/message broker service (e.g., Redis).
4.  **Install System Utilities:** Ensure `ExifTool` is installed on the metadata commitment node (5090 PC).

## 📄 License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.
