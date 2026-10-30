"""
ISL alphabet (A-Z) real-time translator - FastAPI backend.

Run:  uvicorn main:app --host 0.0.0.0 --port 8000
Expects:  ./model/keras_model.h5  and  ./model/labels.txt  (Google Teachable Machine export)

WebSocket protocol  (ws://localhost:8000/ws)
  client -> server : {"frame": "<base64 JPEG, with or without data-URL prefix>"}
  server -> client : {
        "status": "ok" | "no_hand" | "uncertain" | "error",
        "prediction": "A" | "",
        "confidence": 0.0-1.0,
        "bbox": [x, y, w, h] | null,   # normalised 0-1, relative to the received frame
        "message": "..."               # only for status == "error"
  }
The client sends one frame, then waits for the reply before sending the next
(simple back-pressure), so the server never builds up a queue of stale frames.
"""
import asyncio
import base64
import binascii
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("isl")

# --------------------------------------------------------------------------- #
# Configuration (override via environment variables)
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "model" / "keras_model.h5"))
LABELS_PATH = Path(os.getenv("LABELS_PATH", BASE_DIR / "model" / "labels.txt"))

MODEL_INPUT_SIZE = 224          # Teachable Machine image models expect 224x224
BBOX_PADDING = 20               # pixels added around the hand box
# ISL alphabet is largely TWO-handed, so track up to two hands and use their union box.
MAX_HANDS = int(os.getenv("MAX_HANDS", "2"))
DETECTION_CONFIDENCE = 0.7
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))

# Teachable Machine's webcam capture is mirrored by default ("Flip" option), so the
# training images are usually mirrored. If you trained with flip OFF, set MIRROR_INPUT=0.
MIRROR_INPUT = os.getenv("MIRROR_INPUT", "1") == "1"

# True  -> keep the crop's aspect ratio and pad to a square on a white canvas (no distortion)
# False -> plain squash resize to 224x224 (matches what TM does with raw webcam frames)
LETTERBOX = os.getenv("LETTERBOX", "1") == "1"

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")


# --------------------------------------------------------------------------- #
# Model holder (loaded once at startup, shared by all connections)
# --------------------------------------------------------------------------- #
class ModelBundle:
    def __init__(self) -> None:
        self.classifier: Optional[Classifier] = None
        self.labels: list[str] = []
        # Keras predict() is not safe to call from several threads at once.
        self.lock = threading.Lock()

    def load(self) -> None:
        for p in (MODEL_PATH, LABELS_PATH):
            if not p.exists():
                raise FileNotFoundError(f"Required model file not found: {p}")
        self.labels = self._read_labels(LABELS_PATH)
        # cvzone's Classifier resizes to 224x224 and normalises to [-1, 1] like Teachable Machine.
        self.classifier = Classifier(str(MODEL_PATH))
        log.info("Loaded model with %d labels: %s", len(self.labels), self.labels)

    @staticmethod
    def _read_labels(path: Path) -> list[str]:
        """Teachable Machine labels.txt lines look like '0 A' - strip the leading index."""
        labels: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            labels.append(parts[1] if len(parts) == 2 and parts[0].isdigit() else line)
        return labels


models = ModelBundle()


@asynccontextmanager
async def lifespan(_: FastAPI):
    models.load()  # fail fast with a clear error if files are missing
    yield


app = FastAPI(title="ISL Alphabet Translator", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "labels": len(models.labels)}


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
def decode_base64_frame(data: str) -> Optional[np.ndarray]:
    """base64 (optionally 'data:image/jpeg;base64,...') -> BGR OpenCV image, or None if invalid."""
    if "," in data[:100]:  # strip the data-URL prefix if the browser included it
        data = data.split(",", 1)[1]
    try:
        raw = base64.b64decode(data, validate=False)
    except (binascii.Error, ValueError):
        return None
    if not raw:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)  # returns None on corrupt JPEG


def union_bbox(hands: list[dict]) -> tuple[int, int, int, int]:
    """Combine the bounding boxes of all detected hands into one box (x, y, w, h)."""
    x1 = min(h["bbox"][0] for h in hands)
    y1 = min(h["bbox"][1] for h in hands)
    x2 = max(h["bbox"][0] + h["bbox"][2] for h in hands)
    y2 = max(h["bbox"][1] + h["bbox"][3] for h in hands)
    return x1, y1, x2 - x1, y2 - y1


def crop_with_padding(frame: np.ndarray, bbox: tuple[int, int, int, int], pad: int):
    """
    Crop bbox + padding, CLAMPED to the frame so slicing never goes negative / out of range.
    (A negative start index in numpy silently wraps around and returns a wrong/empty crop,
    which is the classic cvzone crash - clamping prevents it.)
    Returns (crop, (x1, y1, x2, y2)) or (None, None) if the clamped region is empty.
    """
    fh, fw = frame.shape[:2]
    x, y, w, h = bbox
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(fw, x + w + pad)
    y2 = min(fh, y + h + pad)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None, None
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def to_model_input(crop: np.ndarray, size: int = MODEL_INPUT_SIZE) -> np.ndarray:
    """Resize the crop to size x size, optionally preserving aspect ratio on a white canvas."""
    if not LETTERBOX:
        return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    h, w = crop.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    y0, x0 = (size - new_h) // 2, (size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized  # centre the crop
    return canvas


# --------------------------------------------------------------------------- #
# Core pipeline: base64 -> frame -> hand(s) -> crop -> classify -> payload
# Runs in a worker thread (blocking CV/TF work must not stall the event loop).
# --------------------------------------------------------------------------- #
def process_frame(b64: str, detector: HandDetector) -> dict:
    frame = decode_base64_frame(b64)
    if frame is None:
        return {"status": "error", "prediction": "", "confidence": 0.0, "bbox": None,
                "message": "Could not decode frame"}

    if MIRROR_INPUT:
        frame = cv2.flip(frame, 1)  # horizontal flip to match mirrored training data

    fh, fw = frame.shape[:2]

    # draw=False: we only need landmarks/bbox, not the overlay drawn onto the frame.
    hands, _ = detector.findHands(frame, draw=False)
    if not hands:
        return {"status": "no_hand", "prediction": "", "confidence": 0.0, "bbox": None}

    bbox = union_bbox(hands)
    crop, (x1, y1, x2, y2) = crop_with_padding(frame, bbox, BBOX_PADDING)
    if crop is None:  # hand sliver at the frame edge -> treat as "no usable hand"
        return {"status": "no_hand", "prediction": "", "confidence": 0.0, "bbox": None}

    model_input = to_model_input(crop)

    with models.lock:
        probs, index = models.classifier.getPrediction(model_input, draw=False)

    confidence = float(probs[index])
    # Normalised box so the frontend can overlay it regardless of display size.
    # If the frame was mirrored here, un-mirror x so it lines up with the raw video.
    nx, nw = x1 / fw, (x2 - x1) / fw
    if MIRROR_INPUT:
        nx = 1.0 - nx - nw
    norm_bbox = [round(nx, 4), round(y1 / fh, 4), round(nw, 4), round((y2 - y1) / fh, 4)]

    label = models.labels[index] if 0 <= index < len(models.labels) else "?"
    status = "ok" if confidence >= CONFIDENCE_THRESHOLD else "uncertain"
    return {
        "status": status,
        "prediction": label if status == "ok" else "",
        "confidence": round(confidence, 4),
        "bbox": norm_bbox,
    }


# --------------------------------------------------------------------------- #
# WebSocket endpoint
# --------------------------------------------------------------------------- #
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    log.info("Client connected: %s", websocket.client)

    # One detector per connection: MediaPipe tracking keeps per-stream state between frames.
    detector = HandDetector(staticMode=False, maxHands=MAX_HANDS, detectionCon=DETECTION_CONFIDENCE)

    try:
        while True:
            text = await websocket.receive_text()

            try:
                frame_b64 = json.loads(text)["frame"]
                if not isinstance(frame_b64, str):
                    raise TypeError
            except (json.JSONDecodeError, KeyError, TypeError):
                await websocket.send_json({"status": "error", "prediction": "", "confidence": 0.0,
                                           "bbox": None, "message": "Malformed message"})
                continue

            try:
                result = await asyncio.to_thread(process_frame, frame_b64, detector)
            except Exception:  # never let one bad frame kill the socket
                log.exception("Frame processing failed")
                result = {"status": "error", "prediction": "", "confidence": 0.0, "bbox": None,
                          "message": "Processing error"}

            await websocket.send_json(result)

    except WebSocketDisconnect:
        log.info("Client disconnected: %s", websocket.client)
    except Exception:
        log.exception("WebSocket error")
    finally:
        try:
            detector.hands.close()  # release MediaPipe resources
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
