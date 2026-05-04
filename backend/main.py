import io
import cv2
import base64
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import json

from pipeline import simulate_pipeline, restore_pipeline

app = FastAPI(
    title="Sea-Thru API",
    description="Underwater image simulation and restoration based on the Sea-Thru algorithm.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def decode_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Please upload a valid image file.")
    return img


def encode_image(img: np.ndarray, fmt: str = ".jpg") -> bytes:
    success, buffer = cv2.imencode(fmt, img)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode output image.")
    return buffer.tobytes()


def img_to_base64(img: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode("utf-8")


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/simulate/image")
async def simulate_image(
    file: UploadFile = File(...),
    params: Optional[str] = Form(None)
):
    """
    Upload a clean image and receive a simulated underwater version.
    Optionally pass filter parameters as a JSON string.
    """
    raw = await file.read()
    img = decode_image(raw)

    p = json.loads(params) if params else {}
    result = simulate_pipeline(img, params=p)

    return JSONResponse({
        "original": img_to_base64(img),
        "result": img_to_base64(result),
        "mode": "simulate"
    })


@app.post("/restore/image")
async def restore_image(
    file: UploadFile = File(...),
    params: Optional[str] = Form(None)
):
    """
    Upload a degraded underwater image and receive a restored version.
    Optionally pass restoration parameters as a JSON string.
    """
    raw = await file.read()
    img = decode_image(raw)

    p = json.loads(params) if params else {}
    result = restore_pipeline(img, params=p)

    return JSONResponse({
        "original": img_to_base64(img),
        "result": img_to_base64(result),
        "mode": "restore"
    })


@app.post("/webcam/frame")
async def process_webcam_frame(
    file: UploadFile = File(...),
    mode: str = Form("simulate"),
    params: Optional[str] = Form(None)
):
    """
    Process a single webcam frame. Mode: 'simulate' or 'restore'.
    Returns base64 encoded processed frame for live preview.
    """
    raw = await file.read()
    img = decode_image(raw)

    p = json.loads(params) if params else {}

    if mode == "simulate":
        result = simulate_pipeline(img, params=p, frame_seed=42)
    elif mode == "restore":
        result = restore_pipeline(img, params=p)
    else:
        raise HTTPException(status_code=400, detail="mode must be 'simulate' or 'restore'")

    return JSONResponse({"result": img_to_base64(result)})


@app.post("/simulate/video")
async def simulate_video(file: UploadFile = File(...)):
    """
    Upload a video file and receive a simulated underwater version.
    Returns the processed video as an mp4 stream.
    """
    raw = await file.read()
    tmp_in = "/tmp/input_video.mp4"
    tmp_out = "/tmp/output_video.mp4"

    with open(tmp_in, "wb") as f:
        f.write(raw)

    cap = cv2.VideoCapture(tmp_in)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    out = cv2.VideoWriter(tmp_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed = simulate_pipeline(frame, frame_seed=frame_idx)
        out.write(processed)
        frame_idx += 1

    cap.release()
    out.release()

    def iter_file():
        with open(tmp_out, "rb") as f:
            yield from f

    return StreamingResponse(iter_file(), media_type="video/mp4")


@app.post("/restore/video")
async def restore_video(file: UploadFile = File(...)):
    """
    Upload a degraded underwater video and receive a restored version.
    """
    raw = await file.read()
    tmp_in = "/tmp/input_video_restore.mp4"
    tmp_out = "/tmp/output_video_restore.mp4"

    with open(tmp_in, "wb") as f:
        f.write(raw)

    cap = cv2.VideoCapture(tmp_in)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    out = cv2.VideoWriter(tmp_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed = restore_pipeline(frame)
        out.write(processed)

    cap.release()
    out.release()

    def iter_file():
        with open(tmp_out, "rb") as f:
            yield from f

    return StreamingResponse(iter_file(), media_type="video/mp4")