# ISL Alphabet Translator (local MVP)

Webcam -> React -> WebSocket -> FastAPI (cvzone hand detection + Teachable Machine classifier) -> live text.

## 1. Backend (Python 3.10 or 3.11)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Put your Teachable Machine export in backend/model/:
#   keras_model.h5
#   labels.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Optional environment variables: `MIRROR_INPUT` (1/0), `LETTERBOX` (1/0), `MAX_HANDS` (1/2),
`CONFIDENCE_THRESHOLD` (default 0.70), `ALLOWED_ORIGINS`.

## 2. Frontend (Node 18+)

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 and allow camera access.
To point at a different backend, set `VITE_WS_URL=ws://host:8000/ws`.

## Notes
- TensorFlow is pinned to 2.15.1 because Teachable Machine .h5 files fail on Keras 3.
- Teachable Machine trains on mirrored webcam frames by default, so the backend flips input (`MIRROR_INPUT=1`).
  If you trained with flip off, set `MIRROR_INPUT=0`.
- Most ISL letters use two hands, so up to two hands are tracked and their combined box is classified.
