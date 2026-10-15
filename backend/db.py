"""
db.py
Gestion de la base de données SQLite : sessions d'analyse et détections.
"""
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional

DB_NAME = "detections.db"


def get_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Crée les tables si elles n'existent pas déjà."""
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_name TEXT,
            task TEXT,                 -- 'detect', 'segment', 'track'
            model_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            frame_index INTEGER,
            track_id INTEGER,
            class_name TEXT,
            confidence REAL,
            bbox_x1 REAL,
            bbox_y1 REAL,
            bbox_x2 REAL,
            bbox_y2 REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        )
        """
    )
    conn.commit()
    conn.close()


def create_session(source_name: str, task: str, model_name: str) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO sessions (source_name, task, model_name) VALUES (?, ?, ?)",
        (source_name, task, model_name),
    )
    session_id = cur.lastrowid
    conn.commit()
    conn.close()
    return session_id


def insert_detections(session_id: int, frame_index: int, detections: List[Dict]):
    """
    detections: liste de dicts avec les clés
    track_id, class_name, confidence, bbox (x1, y1, x2, y2)
    """
    if not detections:
        return
    conn = get_connection()
    conn.executemany(
        """
        INSERT INTO detections
        (session_id, frame_index, track_id, class_name, confidence,
         bbox_x1, bbox_y1, bbox_x2, bbox_y2)
        VALUES (:session_id, :frame_index, :track_id, :class_name, :confidence,
                :bbox_x1, :bbox_y1, :bbox_x2, :bbox_y2)
        """,
        [
            {
                "session_id": session_id,
                "frame_index": frame_index,
                "track_id": d.get("track_id"),
                "class_name": d["class_name"],
                "confidence": d["confidence"],
                "bbox_x1": d["bbox"][0],
                "bbox_y1": d["bbox"][1],
                "bbox_x2": d["bbox"][2],
                "bbox_y2": d["bbox"][3],
            }
            for d in detections
        ],
    )
    conn.commit()
    conn.close()


def get_sessions() -> List[Dict]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM sessions ORDER BY created_at DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_session_stats(session_id: int) -> Dict:
    """Statistiques agrégées pour une session : nb détections, classes, tracks uniques."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT COUNT(*) as total FROM detections WHERE session_id = ?", (session_id,)
    )
    total = cur.fetchone()["total"]

    cur.execute(
        """
        SELECT class_name, COUNT(*) as count
        FROM detections WHERE session_id = ?
        GROUP BY class_name ORDER BY count DESC
        """,
        (session_id,),
    )
    by_class = [dict(r) for r in cur.fetchall()]

    cur.execute(
        """
        SELECT COUNT(DISTINCT track_id) as unique_tracks
        FROM detections WHERE session_id = ? AND track_id IS NOT NULL
        """,
        (session_id,),
    )
    unique_tracks = cur.fetchone()["unique_tracks"]

    conn.close()
    return {"total_detections": total, "by_class": by_class, "unique_tracks": unique_tracks}


def get_detections_for_session(session_id: int) -> List[Dict]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM detections WHERE session_id = ? ORDER BY frame_index",
        (session_id,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows
