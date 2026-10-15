"""
main.py
API FastAPI : expose des endpoints pour la détection, la segmentation et le
suivi d'objets (tracking) sur images/vidéos, avec stockage des résultats
en base SQLite.
"""
import os
import shutil
import uuid
import logging

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse

import db
import detector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vision-api")

app = FastAPI(title="Vision Detection/Tracking API", version="1.0")

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.on_event("startup")
def on_startup():
    db.init_db()
    logger.info("Base de données initialisée.")


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Détection sur image
# ---------------------------------------------------------------------------
@app.post("/detect")
async def detect_image(
    file: UploadFile = File(...),
    model_name: str = "yolov8n.pt",
    conf: float = 0.25,
):
    image_bytes = await file.read()
    npimg = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Image invalide.")

    result = detector.run_detection(frame, model_name=model_name, conf=conf)

    session_id = db.create_session(file.filename, "detect", model_name)
    db.insert_detections(session_id, 0, result["detections"])

    output_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}.jpg")
    cv2.imwrite(output_path, result["annotated_frame"])

    return {
        "session_id": session_id,
        "detections": result["detections"],
        "counts": detector.count_by_class(result["detections"]),
        "annotated_image_path": output_path,
    }


# ---------------------------------------------------------------------------
# Segmentation sur image
# ---------------------------------------------------------------------------
@app.post("/segment")
async def segment_image(
    file: UploadFile = File(...),
    model_name: str = "yolov8n-seg.pt",
    conf: float = 0.25,
):
    image_bytes = await file.read()
    npimg = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Image invalide.")

    result = detector.run_segmentation(frame, model_name=model_name, conf=conf)

    session_id = db.create_session(file.filename, "segment", model_name)
    db.insert_detections(session_id, 0, result["detections"])

    output_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}.jpg")
    cv2.imwrite(output_path, result["annotated_frame"])

    return {
        "session_id": session_id,
        "detections": result["detections"],
        "counts": detector.count_by_class(result["detections"]),
        "annotated_image_path": output_path,
    }


# ---------------------------------------------------------------------------
# Tracking sur vidéo (traite toutes les frames et stocke les résultats)
# ---------------------------------------------------------------------------
@app.post("/track-video")
async def track_video(
    file: UploadFile = File(...),
    model_name: str = "yolov8n.pt",
    conf: float = 0.25,
    tracker: str = "bytetrack.yaml",
    frame_skip: int = 1,
):
    """
    Traite une vidéo entière : applique la détection + le tracking frame par
    frame, enregistre chaque détection en base, et produit une vidéo annotée.
    `frame_skip` permet de ne traiter qu'une frame sur N pour accélérer.
    """
    input_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Vidéo illisible.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    session_id = db.create_session(file.filename, "track", model_name)

    frame_index = 0
    processed_frames = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_skip == 0:
                result = detector.run_tracking_on_frame(
                    frame, model_name=model_name, conf=conf, tracker=tracker
                )
                db.insert_detections(session_id, frame_index, result["detections"])
                writer.write(result["annotated_frame"])
                processed_frames += 1
            else:
                writer.write(frame)

            frame_index += 1
    finally:
        cap.release()
        writer.release()

    stats = db.get_session_stats(session_id)

    return {
        "session_id": session_id,
        "total_frames": frame_index,
        "processed_frames": processed_frames,
        "annotated_video_path": output_path,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# Récupération de fichiers résultat (image/vidéo annotée)
# ---------------------------------------------------------------------------
@app.get("/media/{filename}")
def get_media(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Fichier introuvable.")
    return FileResponse(path)


# ---------------------------------------------------------------------------
# Sessions & statistiques (pour le dashboard)
# ---------------------------------------------------------------------------
@app.get("/sessions")
def list_sessions():
    return db.get_sessions()


@app.get("/sessions/{session_id}/stats")
def session_stats(session_id: int):
    return db.get_session_stats(session_id)


@app.get("/sessions/{session_id}/detections")
def session_detections(session_id: int):
    return db.get_detections_for_session(session_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
