# Digital Inspector – Backend

FastAPI service that powers the ARMETA Digital Inspector demo. The API accepts
PDF/images, runs three YOLOv8 models (stamps, QR codes, signatures) and returns
structured detections together with annotated previews.

---

## Project Layout

```
backend/
├─ app/
│  ├─ main.py          ← FastAPI application entrypoint
│  ├─ inference.py     ← YOLO model loading and inference helpers
│  ├─ pdf_utils.py     ← PDF → image conversion utilities
│  └─ preprocess.py    ← Reserved for additional pre-processing
├─ requirements.txt    ← Python dependencies
├─ Dockerfile          ← Production image definition
└─ README.md           ← This file
```

---

## Prerequisites

- **Python 3.11+**
- **pip** (or uv / poetry if you prefer)
- **Windows / WSL / Linux / macOS** – the service is OS-agnostic
- Optional but recommended: a GPU with CUDA support for faster YOLO inference

Model checkpoints must exist in the repository root:

| File           | Purpose        | Default env variable                |
| -------------- | -------------- | ----------------------------------- |
| `best.pt`      | Stamp detector | `STAMP_MODEL_PATH`                  |
| `best_qr.pt`   | QR detector    | `QR_MODEL_PATH` (optional)          |
| `best_sign.pt` | Signature det. | `SIGNATURE_MODEL_PATH` (optional)   |

If a file is missing you will need to supply the absolute path via the
corresponding environment variable before launching the API.

---

## Installation

```powershell
cd D:\ARMETA\backend

# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # PowerShell
# source .venv/bin/activate     # bash / WSL

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

If you plan to run the service inside Docker:

```powershell
docker build -t digital-inspector-backend .
```

---

## Configuration

Environment variables (place them in `.env` or export before running):

| Variable               | Description | Default |
| ---------------------- | ----------- | ------- |
| `STAMP_MODEL_PATH`     | Absolute/relative path to `best.pt` | `<repo_root>/best.pt` |
| `QR_MODEL_PATH`        | Path to `best_qr.pt` (optional)     | `<repo_root>/best_qr.pt` |
| `SIGNATURE_MODEL_PATH` | Path to `best_sign.pt` (optional)   | `<repo_root>/best_sign.pt` |
| `CORS_ORIGINS`         | Comma-separated allowed origins     | `http://localhost:5173` |

Multiple values for `CORS_ORIGINS` should be separated with commas, e.g.
`http://localhost:5173,https://armeta.io`.

---

## Running the API

### Development (auto reload)

```powershell
cd D:\ARMETA\backend
.venv\Scripts\activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production (example)

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
# or docker run -p 8000:8000 digital-inspector-backend
```

The service will preload all configured YOLO models at startup. If a model file
is missing, startup will fail with a clear error message.

---

## API Endpoints

| Method | Path                    | Description |
| ------ | ----------------------- | ----------- |
| GET    | `/health`               | Liveness probe, returns `{"status":"ok"}` |
| POST   | `/api/analyze`          | Accepts one or more documents and returns detections per page (without annotated images). |
| POST   | `/api/analyze/annotated`| Same as above but includes base64 annotated previews. Pages are processed in parallel and previews are JPEG-compressed to reduce payload size. |

### Example request (PowerShell)

```powershell
Invoke-WebRequest `
  -Uri http://localhost:8000/api/analyze/annotated `
  -Method POST `
  -Form @{ files = Get-Item "sample.pdf" }
```

Response structure:

```json
{
  "files": [
    {
      "file_name": "sample.pdf",
      "pages": [
        {
          "page_index": 0,
          "width": 2480,
          "height": 3508,
          "objects": [
            { "class": "stamp", "confidence": 0.93, "bbox": [120, 450, 380, 380] }
          ],
          "image_base64": "..." // only for /api/analyze/annotated
        }
      ]
    }
  ]
}
```

---

## Performance Notes

- PDFs are rasterized at **200 DPI** for faster turnaround (configurable in
  `app/main.py` if you need higher fidelity).
- Pages are processed concurrently through `ThreadPoolExecutor`, allowing many
  pages/files to be analyzed without blocking the entire event loop.
- Annotated previews are resized (max 1600px) and saved as JPEG (quality 75) to
  minimize the payload returned to the frontend.
- Frontend uploads are automatically batched (10 files at a time) so sending
  dozens of documents will not crash the browser.

For heavy workloads consider running the service with a GPU-enabled build of
PyTorch/Ultralytics to leverage CUDA.

---

## Troubleshooting

| Symptom | Fix |
| ------- | --- |
| `ModuleNotFoundError: fastapi` | Re-run `pip install -r requirements.txt` inside the virtual environment. |
| `FileNotFoundError: best.pt`   | Check that the model files exist in `D:\ARMETA\` or update the `*_MODEL_PATH` env vars. |
| PDFs take too long to render | Lower the DPI in `pdf_utils.pdf_bytes_to_images` or upgrade hardware. |
| Large responses | Use `/api/analyze` (without annotated images) or decrease `max_dimension`/`quality` values in `_encode_image_to_base64`. |

---

## License

Internal ARMETA project – see the root repository for licensing information.
