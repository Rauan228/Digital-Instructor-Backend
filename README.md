# Digital Inspector – Backend

FastAPI service that powers the ARMETA Digital Inspector. It accepts PDFs and images, runs object detection (stamps, signatures, authors, signauth, QR codes) via Ultralytics YOLO or Roboflow Hosted Inference, and returns per-page results. Optionally provides annotated previews and writes convenient batch JSONs for downstream processing.

---

## Tech Stack

- Python `3.10`
- FastAPI + Uvicorn
- Ultralytics YOLO (PyTorch) for local inference
- OpenCV + NumPy for image handling
- PyMuPDF (`pymupdf`) for PDF rendering
- Requests for optional Roboflow Hosted Inference

---

## Project Layout

```
backend/
├─ app/
│  ├─ main.py        ← FastAPI app, endpoints and orchestration
│  ├─ inference.py   ← YOLO / Roboflow inference logic + drawing utils
│  ├─ pdf_utils.py   ← PDF → image conversion (PyMuPDF)
│  └─ preprocess.py  ← (reserved for preprocessing helpers)
├─ requirements.txt
├─ Dockerfile
├─ .env.example
└─ .python-version   ← 3.10

repo-root/
├─ best.pt           ← stamp detector (required)
├─ best_qr.pt        ← QR detector (optional)
├─ sign-auth.pt      ← signatures/auth/signauth detector (preferred)
└─ best_sign.pt      ← signatures/auth fallback if `sign-auth.pt` absent
```

Notes:
- Model weights are expected in the repository root by default. You can override paths via environment variables.
- The service writes annotation JSONs into `selected_output/` (configurable).

---

## Endpoints

- `GET /health`
  - Returns `{ "status": "ok" }`.

- `POST /api/analyze`
  - Request: `multipart/form-data` with one or more files under field `files`.
  - Response: per-file pages with objects only (no previews).
  - Side effects: writes/updates `selected_output/selected_annotations.json`, `selected_output/masked_annotations.json`, and batch variants.

- `POST /api/analyze/annotated`
  - Same as `/api/analyze`, but every page also includes `image_base64` of the annotated preview.

### Request Example (PowerShell)

```powershell
curl -X POST http://localhost:8000/api/analyze \ 
  -F "files=@d:/ARMETA/images/val/Frame_108.jpg" \ 
  -F "files=@d:/ARMETA/images/val/Frame_160.jpg"
```

### Response Shape

```json
{
  "files": [
    {
      "file_name": "Frame_108.jpg",
      "pages": [
        {
          "page_index": 0,
          "width": 1920,
          "height": 1080,
          "objects": [
            { "class": "stamp", "confidence": 0.94, "bbox": [x, y, w, h] },
            { "class": "signature", "confidence": 0.88, "bbox": [x, y, w, h] },
            { "class": "auth", "confidence": 0.77, "bbox": [x, y, w, h] },
            { "class": "signauth", "confidence": 0.65, "bbox": [x, y, w, h] },
            { "class": "qr", "confidence": 0.92, "bbox": [x, y, w, h] }
          ]
          // For /api/analyze/annotated only:
          // "image_base64": "data:image/jpeg;base64,/9j/..."
        }
      ]
    }
  ]
}
```

---

## Exports (Server-Side)

The backend maintains easy-to-consume JSONs in `selected_output/`:

- `selected_annotations.json` — per-file raw classes (`stamp`, `signature`, `qr`, `auth`, `signauth`).
- `masked_annotations.json` — same data with classes mapped to canonical labels:
  - `stamp → label_1`, `signature → label_2`, `qr → label_3`, `auth → label_4`, `signauth → label_5`.
- Batch files (rewritten on each request with files):
  - `selected_annotations_batch.json`
  - `masked_annotations_batch.json`

---

## Installation

### Prerequisites

- Python `3.10` (see `.python-version`)
- Recommended: a virtual environment

### Setup (Windows PowerShell)

```powershell
cd d:\ARMETA\backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Environment Configuration

Create `.env` in `backend/` (or use system env vars). Common options:

```env
# CORS for your frontend(s)
# Comma-separated list of allowed origins
CORS_ORIGINS=http://localhost:5173,http://localhost:5174
# Optional regex for wildcard origins (used together with CORS_ORIGINS)
CORS_ORIGIN_REGEX=https://.*\.vercel\.app

# Confidence threshold for detections (default: 0.25)
DETECT_CONF=0.25

# Optional custom output directory for annotation JSONs
ANNOTATIONS_OUTPUT_DIR=selected_output

# Explicit model weights (otherwise defaults to repo-root files below)
STAMP_MODEL_PATH=d:/ARMETA/best.pt
QR_MODEL_PATH=d:/ARMETA/best_qr.pt
SIGNATURE_MODEL_PATH=d:/ARMETA/sign-auth.pt

# Roboflow Hosted Inference (optional; overrides local signature YOLO)
ROBOFLOW_MODEL_ID=signature-krkm0/1
ROBOFLOW_API_KEY=your_api_key_here
# Optional custom endpoint (if not provided, built from model id)
SIGNATURE_ROBOFLOW_ENDPOINT=https://detect.roboflow.com/signature-krkm0/1?api_key=your_api_key_here&format=json
# Optional tuning
ROBOFLOW_CONF=0.25
ROBOFLOW_OVERLAP=30
ROBOFLOW_FORMAT=json
```

Notes:
- If `SIGNATURE_MODEL_PATH` is not set, the service uses `sign-auth.pt` if present, otherwise `best_sign.pt`.
- If both `ROBOFLOW_MODEL_ID` and `ROBOFLOW_API_KEY` are set, signature detection uses Roboflow instead of local YOLO.
- Legacy `MODEL_PATH` from older configs is ignored by the current code; use `STAMP_MODEL_PATH`, `QR_MODEL_PATH`, and `SIGNATURE_MODEL_PATH` instead.
- `CORS_ORIGIN_REGEX` can be combined with `CORS_ORIGINS` to allow wildcard domains; by default the backend allows `https://.*\.vercel\.app`.

### Run (development)

```powershell
uvicorn app.main:app --reload --port 8000
# Open http://localhost:8000/docs for interactive Swagger UI
```

---

## Docker

Build the image (context is `backend/`):

```powershell
cd d:\ARMETA\backend
docker build -t digital-inspector-backend .
```

Run with volume mounts for weights and output directory:

```powershell
docker run --rm -p 8000:8000 ^
  -e CORS_ORIGINS=http://localhost:5173 ^
  -v ${PWD}\..\best.pt:/app/best.pt ^
  -v ${PWD}\..\best_qr.pt:/app/best_qr.pt ^
  -v ${PWD}\..\sign-auth.pt:/app/sign-auth.pt ^
  -v ${PWD}\..\selected_output:/app/selected_output ^
  digital-inspector-backend
```

Notes:
- The Dockerfile copies only `app/` and `requirements.txt`. Mount weights from the repo root or bake them in a custom image.

---

## How It Works

- PDF handling: `pdf_utils.py` uses PyMuPDF to render pages into JPEG/PNG and passes them to inference.
- Inference:
  - Local YOLO via Ultralytics for `stamp`, `qr`, `signature`, `auth`, `signauth` classes.
  - Optional Roboflow endpoint for signature detection.
- `/api/analyze` renders PDFs at ~300 DPI; `/api/analyze/annotated` uses ~200 DPI and creates compressed annotated previews (`image_base64`).
- Confidence threshold is controlled by `DETECT_CONF`.
- Exports are accumulated and masked into `selected_output/` JSON files.

---

## Troubleshooting

- CORS blocked by browser
  - Set `CORS_ORIGINS` to include your frontend origin(s) and restart the server.

- PDFs fail to process with "PyMuPDF not installed"
  - Ensure `pymupdf` is installed (`requirements.txt` already includes it).

- GPU not used / slow inference
  - Ultralytics/PyTorch will use CPU unless a compatible CUDA build is installed. Install the appropriate `torch` wheel for your GPU, or keep CPU and tune `DETECT_CONF`/DPI.

- Roboflow request timeouts
  - Check connectivity and your `ROBOFLOW_*` settings. Consider using local YOLO by omitting Roboflow vars.

- `FileNotFoundError` for model weights
  - Provide paths via env vars or place files in the repo root (`best.pt`, `best_qr.pt`, `sign-auth.pt` or `best_sign.pt`).

- Large inputs or memory errors
  - Reduce PDF DPI (code-level tweak) or limit page count upstream. Annotated mode already uses a lower DPI.

---

## Security & Notes

- This demo API accepts file uploads and runs heavy processing; do not expose it publicly without rate limits and appropriate hardening.
- Inputs are not stored by default; only aggregated annotations are written to `selected_output/`.
- The API surface is minimal to support the frontend use-case.

---

## License

Internal ARMETA project — see the root repository for licensing information.
