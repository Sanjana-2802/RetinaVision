# Cloud Deployment & PyTorch Package Optimization Guide

This guide explains how to deploy **SpectraRetina** to cloud environments (FastAPI Cloud, Render, Railway, Hugging Face Spaces, Google Cloud Run, AWS, Docker) while solving the **4.5 GB NVIDIA CUDA wheel bloat** issue.

---

## 1. The NVIDIA / CUDA Bloat Problem Explained

When deploying Python apps that list `torch` in `requirements.txt` to Linux cloud environments, **pip installs the default PyPI wheel**, which bundles the entire NVIDIA CUDA runtime:

| Package | Standard Linux Size | Why It Was Being Downloaded |
|---|---|---|
| `nvidia-cudnn-cu12` | ~700 MB | Deep Neural Network GPU acceleration |
| `nvidia-cublas-cu12` | ~500 MB | Basic Linear Algebra Subprograms GPU |
| `nvidia-cusparse-cu12` | ~250 MB | Sparse matrix routines GPU |
| `nvidia-cusolver-cu12` | ~150 MB | Dense/Sparse direct solvers GPU |
| `nvidia-cufft-cu12` | ~150 MB | Fast Fourier Transform GPU |
| `nvidia-nccl-cu12` | ~200 MB | Multi-GPU collective communications |
| `nvidia-cuda-runtime-cu12` | ~10 MB | CUDA runtime driver library |
| Other `nvidia-*` + `triton` | ~500 MB | Compiler and helper packages |
| **Total PyTorch Footprint** | **~4.5 GB** | **Intended for physical NVIDIA GPUs** |

### The Problem on Cloud Hosts:
- 99% of cloud web servers (Render Free/Starter, Railway, Google Cloud Run, Fly.io, Heroku) run on **standard CPU instances** (no physical NVIDIA GPU attached).
- Downloading 4.5 GB causes **disk quota exceeded**, **build timeouts (15+ min)**, and **OOM memory kills**.

---

## 2. The Solution: Official PyTorch CPU-Only Index

PyTorch publishes official **CPU-only builds** without any NVIDIA libraries:
- **Download size**: **~180 MB** (instead of 4,500 MB &mdash; an **85%–95% reduction**!)
- **Install time**: **< 30 seconds** (instead of 15 minutes)
- **Index URL**: `https://download.pytorch.org/whl/cpu`

We have preconfigured four comprehensive layers of protection across the codebase:

1. **`pyproject.toml` (FastAPI Cloud & uv builders)**:
   ```toml
   [[tool.uv.index]]
   name = "pytorch-cpu"
   url = "https://download.pytorch.org/whl/cpu"
   explicit = true

   [tool.uv.sources]
   torch = { index = "pytorch-cpu" }
   torchvision = { index = "pytorch-cpu" }
   ```
   *FastAPI Cloud uses `uv` under the hood; this guarantees that `torch` and `torchvision` resolve strictly to lightweight `+cpu` wheels with zero `nvidia-*` CUDA bloat!*

2. **`requirements.txt` (Standard pip cloud runners)**:
   Added `--extra-index-url https://download.pytorch.org/whl/cpu` at line 1.

3. **`Dockerfile` (Docker / Google Cloud Run / AWS / Hugging Face)**:
   Installs PyTorch CPU explicitly first before installing other packages:
   ```dockerfile
   RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
   RUN pip install --no-cache-dir -r requirements.txt
   ```

4. **`build.sh` (Render / Railway / custom buildpacks)**:
   One-line command for cloud pipelines:
   ```bash
   pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
   ```

---

## 3. Deployment Options

### Option A: FastAPI Cloud (`fastapi deploy`)
The project is configured for 1-command deployment to **FastAPI Cloud** via the official `fastapi-cli`:

1. **Verify you are logged in**:
   ```bash
   fastapi cloud whoami
   # If not logged in:
   fastapi login
   ```

2. **Deploy directly with a single command**:
   ```bash
   fastapi deploy
   ```
   *FastAPI Cloud will automatically read `pyproject.toml`, discover the entrypoint `app.main:app`, route `torch` and `torchvision` directly to the official PyTorch CPU wheel index using `[tool.uv.sources]`, apply `.fastapicloudignore`, and provision your live HTTPS URL (`https://<app-name>.fastapicloud.dev`).*

3. **Check logs & deployment status**:
   ```bash
   fastapi cloud logs
   fastapi cloud deployments
   ```

---


### Option B: Render (1-Click Blueprint or Manual Web Service)

We have provided [`render.yaml`](file:///c:/Users/heman/Desktop/Eye-retina/render.yaml) for automatic configuration.

**If deploying manually on Render dashboard**:
1. Create a **New Web Service** &rarr; Connect your GitHub repository.
2. **Runtime**: Python 3.
3. **Build Command**:
   ```bash
   pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
   ```
   *(or simply `bash build.sh`)*
4. **Start Command**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port $PORT
   ```
5. **Health Check Path**: `/api/health`

---

### Option C: Railway

We have provided [`railway.toml`](file:///c:/Users/heman/Desktop/Eye-retina/railway.toml) and [`Procfile`](file:///c:/Users/heman/Desktop/Eye-retina/Procfile).

1. In Railway Dashboard &rarr; **New Project** &rarr; **Deploy from GitHub repo**.
2. Railway will automatically detect `Procfile` and `railway.toml`.
3. Build command:
   ```bash
   pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
   ```
4. Start command:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port $PORT
   ```

---

### Option D: Docker Container (Google Cloud Run / AWS App Runner / Fly.io / Hugging Face Spaces)

We have updated [`Dockerfile`](file:///c:/Users/heman/Desktop/Eye-retina/Dockerfile) to be cloud-optimized and dynamic-port ready:

```bash
# Build the lightweight image (size is ~600MB total, vs ~5.2GB with CUDA)
docker build -t eye-retina-ai .

# Run locally or in container runtime
docker run -p 8000:8000 -e PORT=8000 eye-retina-ai
```

---

## 4. Verification Check

Once deployed, query the health check endpoint to verify that all 4 clinical engines are loaded and online:

```bash
curl https://your-cloud-app.com/api/health
```

Expected JSON response:
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "four_pillars": {
    "1_multi_disease_screening": "GhostNet (Multi-Label Sigmoid, Grad-CAM Saliency)",
    "2_disease_severity_grading": "ConvNeXt (5-Stage ICDR Hybrid Severity Score)",
    "3_pixel_level_segmentation": "U-Net 4-Channel (MA, HE, EX, SE Binary Masks)",
    "4_landmark_localization": "Optic Disc CDR (Glaucoma) & Fovea Pinpointing (CSME Risk)"
  },
  "public_benchmarks": ["IDRiD", "Messidor-2", "DDR", "REFUGE", "ODIR-5K"]
}
```
