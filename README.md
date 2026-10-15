# Vision Dashboard

A small full-stack project for running object detection, instance segmentation, and multi-object tracking on images and videos, with every result saved so you can look back at past runs.

I built this to get some hands-on experience serving a real computer vision model behind an actual API, instead of just running inference in a notebook — plus a lightweight dashboard on top to actually *use* it without touching curl or Postman every time.

Under the hood it's **YOLOv8** (via Ultralytics) for the detection/segmentation/tracking itself, wrapped in a **FastAPI** backend, with **SQLite** for storage and a **Streamlit** app as the UI.

## What it does

- **Detection** — upload an image, get back bounding boxes, class names, and confidence scores for every object found.
- **Segmentation** — same, but with pixel-level instance masks instead of just boxes.
- **Tracking** — upload a video and it runs detection frame by frame while keeping a stable ID on each object across frames (via ByteTrack), so the same car or person keeps the same track ID from the first frame to the last. Outputs an annotated video.
- **History** — every run gets stored as a "session" in SQLite (source file, task, model used, every individual detection), browsable later from the dashboard without re-running anything.

## Stack

| | |
|---|---|
| Model | YOLOv8 (Ultralytics), pretrained on COCO — no custom training |
| Tracking | ByteTrack |
| Backend | FastAPI + Uvicorn |
| Image/video I/O | OpenCV |
| Storage | SQLite (stdlib `sqlite3`, no ORM) |
| Frontend | Streamlit |

## Project structure

```
vision-tracking-project/
├── backend/
│   ├── main.py        # FastAPI app — /detect, /segment, /track-video, /sessions...
│   ├── detector.py     # YOLOv8 wrapper (detection / segmentation / tracking)
│   ├── db.py           # SQLite helpers (sessions + detections)
│   ├── uploads/         # uploaded videos land here at runtime
│   └── outputs/         # annotated images/videos the API serves back
├── frontend/
│   └── app.py          # Streamlit dashboard
└── requirements.txt
```

## Getting started

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The YOLOv8 weights (`yolov8n.pt`, `yolov8n-seg.pt`, etc.) aren't bundled in this repo — Ultralytics downloads whichever one you ask for automatically the first time it's used, so the first request through a given model will be a bit slower than the rest.

Then, two terminals:

**1. Backend** (from `backend/`):
```bash
cd backend
uvicorn main:app --reload --port 8000
```
Interactive API docs land on `http://localhost:8000/docs`.

**2. Frontend** (from `frontend/`, in a second terminal):
```bash
cd frontend
streamlit run app.py
```
Opens the dashboard at `http://localhost:8501`. It talks to the backend on port 8000 by default — override with the `API_URL` environment variable if you're running the API somewhere else.

## Notes / things I'd still want to improve

- `/detect` and `/segment` are currently handled as two very similar endpoints — could collapse into one shared code path.
- Segmentation masks get drawn on the output image but aren't persisted anywhere structured (only boxes are saved) — would be worth storing if I ever need the raw mask data later.
- Video processing is fully synchronous right now: a long video just blocks the request until it's done, with a spinner and no real progress bar. A background job queue would fix that properly.
- No auth on the API — fine for running locally, not something I'd expose publicly as-is.

Other directions I'd like to explore at some point: swapping ByteTrack for DeepSORT/StrongSORT to compare tracking quality, adding line-crossing counting, fine-tuning on a custom dataset instead of stock COCO classes, and wrapping the whole thing in Docker.

## License

MIT — see [LICENSE](LICENSE).
