"""
detector.py
Encapsule les modèles YOLOv8 (Ultralytics) pour la détection, la segmentation
et le suivi (tracking) d'objets, image par image ou frame de vidéo.
"""
from typing import List, Dict, Optional
import numpy as np
from ultralytics import YOLO

# Cache des modèles déjà chargés pour éviter de les recharger à chaque appel
_MODEL_CACHE: Dict[str, YOLO] = {}


def load_model(model_name: str = "yolov8n.pt") -> YOLO:
    """Charge (ou récupère depuis le cache) un modèle YOLOv8."""
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = YOLO(model_name)
    return _MODEL_CACHE[model_name]


def _result_to_detections(result) -> List[Dict]:
    """Convertit un objet Result d'Ultralytics en liste de dicts simples."""
    detections = []
    if result.boxes is None:
        return detections

    boxes = result.boxes
    names = result.names

    xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.array([])
    conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.array([])
    cls = boxes.cls.cpu().numpy() if boxes.cls is not None else np.array([])
    track_ids = (
        boxes.id.cpu().numpy() if getattr(boxes, "id", None) is not None else None
    )

    for i in range(len(xyxy)):
        detections.append(
            {
                "class_name": names[int(cls[i])],
                "confidence": float(conf[i]),
                "bbox": [float(v) for v in xyxy[i]],
                "track_id": int(track_ids[i]) if track_ids is not None else None,
            }
        )
    return detections


def run_detection(image, model_name: str = "yolov8n.pt", conf: float = 0.25) -> Dict:
    """Détection d'objets sur une image unique (numpy array BGR ou chemin fichier)."""
    model = load_model(model_name)
    results = model.predict(image, conf=conf, verbose=False)
    result = results[0]
    return {
        "detections": _result_to_detections(result),
        "annotated_frame": result.plot(),  # image numpy BGR annotée
    }


def run_segmentation(image, model_name: str = "yolov8n-seg.pt", conf: float = 0.25) -> Dict:
    """Segmentation d'instance sur une image unique."""
    model = load_model(model_name)
    results = model.predict(image, conf=conf, verbose=False)
    result = results[0]
    return {
        "detections": _result_to_detections(result),
        "annotated_frame": result.plot(),
    }


def run_tracking_on_frame(
    frame,
    model_name: str = "yolov8n.pt",
    conf: float = 0.25,
    tracker: str = "bytetrack.yaml",
    persist: bool = True,
) -> Dict:
    """
    Tracking d'objets sur une frame de vidéo (flux continu).
    `persist=True` conserve l'état du tracker entre les appels successifs
    sur les frames d'une même vidéo/flux.
    """
    model = load_model(model_name)
    results = model.track(
        frame, conf=conf, tracker=tracker, persist=persist, verbose=False
    )
    result = results[0]
    return {
        "detections": _result_to_detections(result),
        "annotated_frame": result.plot(),
    }


def count_by_class(detections: List[Dict]) -> Dict[str, int]:
    """Petit utilitaire : compte le nombre de détections par classe pour une frame."""
    counts: Dict[str, int] = {}
    for d in detections:
        counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
    return counts
