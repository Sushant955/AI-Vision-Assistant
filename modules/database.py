"""SQLite history of detections."""
import os
import sqlite3
from datetime import datetime

import pandas as pd

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database", "history.db")


def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS detections (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               timestamp TEXT NOT NULL,
               source TEXT NOT NULL,
               label TEXT NOT NULL,
               confidence REAL NOT NULL)"""
    )
    return conn


def log_detections(source, detections):
    """detections = list of {'label':..., 'confidence':...}"""
    if not detections:
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [(ts, source, d["label"], float(d["confidence"])) for d in detections]
    with _conn() as conn:
        conn.executemany(
            "INSERT INTO detections (timestamp, source, label, confidence) VALUES (?,?,?,?)", rows
        )


def get_history(limit=2000):
    with _conn() as conn:
        return pd.read_sql_query(
            "SELECT timestamp, source, label, confidence FROM detections ORDER BY id DESC LIMIT ?",
            conn, params=(limit,),
        )


def label_counts():
    with _conn() as conn:
        return pd.read_sql_query(
            "SELECT label, COUNT(*) AS count FROM detections GROUP BY label ORDER BY count DESC", conn
        )


def totals():
    with _conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        labels = conn.execute("SELECT COUNT(DISTINCT label) FROM detections").fetchone()[0]
        avg = conn.execute("SELECT AVG(confidence) FROM detections").fetchone()[0]
    return {"total": total, "labels": labels, "avg_conf": round(avg, 1) if avg else 0.0}


def clear():
    with _conn() as conn:
        conn.execute("DELETE FROM detections")
