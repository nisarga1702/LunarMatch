"""Crater Catalog — SQLite-backed crater database for LunarMatch-TGS.
Stores numbered craters with reference image patches, coordinates, and metadata.
Auto-initializes with pre-built famous craters on first run."""

import sqlite3
import json
import os
import io
import cv2
import numpy as np
from datetime import datetime

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "crater_catalog.db")


def _ensure_db():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS craters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            label TEXT,
            latitude REAL,
            longitude REAL,
            diameter_km REAL,
            radius_px INTEGER,
            center_x INTEGER,
            center_y INTEGER,
            source TEXT DEFAULT 'detected',
            created_at TEXT,
            metadata TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS crater_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            crater_id INTEGER NOT NULL,
            image_blob BLOB NOT NULL,
            image_type TEXT DEFAULT 'reference',
            sun_azimuth REAL,
            sun_elevation REAL,
            sensor TEXT,
            confirmed INTEGER DEFAULT 0,
            added_at TEXT,
            FOREIGN KEY (crater_id) REFERENCES craters(id)
        )
    """)
    conn.commit()
    return conn


def _img_to_blob(img):
    """Encode image as PNG bytes."""
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes() if ok else b''


def _blob_to_img(blob):
    """Decode PNG bytes to image."""
    arr = np.frombuffer(blob, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ──────────────────────────────────────────
# CRUD operations
# ──────────────────────────────────────────

def add_crater(name, label, lat=0.0, lon=0.0, diameter_km=0.0,
               radius_px=0, center_x=0, center_y=0, source='detected', metadata=None):
    """Insert a new crater. Returns the crater_id."""
    conn = _ensure_db()
    cur = conn.execute("""
        INSERT INTO craters (name, label, latitude, longitude, diameter_km,
                             radius_px, center_x, center_y, source, created_at, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (name, label, lat, lon, diameter_km, radius_px, center_x, center_y,
          source, datetime.utcnow().isoformat(), json.dumps(metadata or {})))
    conn.commit()
    cid = cur.lastrowid
    conn.close()
    return cid


def add_crater_image(crater_id, image, image_type='reference',
                     sun_az=0.0, sun_el=0.0, sensor='LROC NAC', confirmed=True):
    """Add a reference or matched image to a crater."""
    conn = _ensure_db()
    conn.execute("""
        INSERT INTO crater_images (crater_id, image_blob, image_type,
                                   sun_azimuth, sun_elevation, sensor, confirmed, added_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (crater_id, _img_to_blob(image), image_type,
          sun_az, sun_el, sensor, 1 if confirmed else 0,
          datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


def get_all_craters():
    """Return list of all craters as dicts."""
    conn = _ensure_db()
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM craters ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_crater(crater_id):
    """Return a single crater dict or None."""
    conn = _ensure_db()
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM craters WHERE id = ?", (crater_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_crater_images(crater_id, image_type=None):
    """Return list of images for a crater. Each dict has 'image' (numpy) + metadata."""
    conn = _ensure_db()
    conn.row_factory = sqlite3.Row
    if image_type:
        rows = conn.execute(
            "SELECT * FROM crater_images WHERE crater_id = ? AND image_type = ?",
            (crater_id, image_type)).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM crater_images WHERE crater_id = ?",
            (crater_id,)).fetchall()
    conn.close()
    results = []
    for r in rows:
        d = dict(r)
        d['image'] = _blob_to_img(d.pop('image_blob'))
        results.append(d)
    return results


def get_catalog_stats():
    """Return summary statistics."""
    conn = _ensure_db()
    total_craters = conn.execute("SELECT COUNT(*) FROM craters").fetchone()[0]
    total_images = conn.execute("SELECT COUNT(*) FROM crater_images").fetchone()[0]
    confirmed = conn.execute("SELECT COUNT(*) FROM crater_images WHERE confirmed = 1").fetchone()[0]
    pending = conn.execute("SELECT COUNT(*) FROM crater_images WHERE confirmed = 0").fetchone()[0]
    conn.close()
    return {
        'total_craters': total_craters,
        'total_images': total_images,
        'confirmed_images': confirmed,
        'pending_images': pending
    }


def confirm_image(image_id):
    """Mark a matched image as confirmed."""
    conn = _ensure_db()
    conn.execute("UPDATE crater_images SET confirmed = 1 WHERE id = ?", (image_id,))
    conn.commit()
    conn.close()


def reject_image(image_id):
    """Remove a rejected match."""
    conn = _ensure_db()
    conn.execute("DELETE FROM crater_images WHERE id = ?", (image_id,))
    conn.commit()
    conn.close()


def delete_crater(crater_id):
    """Delete a crater and all its images."""
    conn = _ensure_db()
    conn.execute("DELETE FROM crater_images WHERE crater_id = ?", (crater_id,))
    conn.execute("DELETE FROM craters WHERE id = ?", (crater_id,))
    conn.commit()
    conn.close()


def catalog_is_empty():
    """Check if catalog has any craters."""
    conn = _ensure_db()
    count = conn.execute("SELECT COUNT(*) FROM craters").fetchone()[0]
    conn.close()
    return count == 0


def get_reference_patches():
    """Return list of (crater_id, crater_name, label, image) for all reference images.
    Used by the matching pipeline to compare against."""
    conn = _ensure_db()
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT c.id as crater_id, c.name, c.label, ci.image_blob,
               ci.sun_azimuth, ci.sun_elevation
        FROM craters c
        JOIN crater_images ci ON c.id = ci.crater_id
        WHERE ci.image_type = 'reference' AND ci.confirmed = 1
        ORDER BY c.id
    """).fetchall()
    conn.close()
    results = []
    for r in rows:
        d = dict(r)
        d['image'] = _blob_to_img(d.pop('image_blob'))
        results.append(d)
    return results
