from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import requests
import os
import random
from typing import List, Optional
import json
from datetime import datetime
import sqlite3
from dotenv import load_dotenv

load_dotenv()

# ── Groq API key loaded from .env — never sent from the browser ───────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

app = FastAPI(title="Road Damage Detection API", version="2.0.0")

# ── CORS: restrict to your actual frontend origin in production ───────────────
# Change "http://localhost:8000" to your deployed domain if you host this online.
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)

# ── Load Model ──────────────────────────────────────────────────────────────
MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Model file not found at {MODEL_PATH}. "
        "Place best.pt in the same directory before starting the server."
    )

model = YOLO(MODEL_PATH)

CLASS_NAMES = ['Alligator', 'Edge Cracking', 'Lateral-Crack',
               'Longitudinal-Crack', 'Ravelling', 'Rutting', 'Striping', 'Pothole']

REPAIR_INFO = {
    'Alligator':          {"cost": "$3,000–$8,000", "method": "full-depth reclamation or overlay", "urgency": "High"},
    'Edge Cracking':      {"cost": "$500–$2,000",   "method": "edge sealing and shoulder repair",  "urgency": "Medium"},
    'Lateral-Crack':      {"cost": "$200–$800",     "method": "crack sealing",                     "urgency": "Low"},
    'Longitudinal-Crack': {"cost": "$200–$800",     "method": "crack sealing",                     "urgency": "Low"},
    'Ravelling':          {"cost": "$1,500–$5,000", "method": "chip seal or surface treatment",    "urgency": "Medium"},
    'Rutting':            {"cost": "$2,000–$6,000", "method": "milling and resurfacing",           "urgency": "High"},
    'Striping':           {"cost": "$100–$400",     "method": "remarking / repainting",            "urgency": "Low"},
    'Pothole':            {"cost": "$50–$500 each", "method": "hot-mix asphalt patching",          "urgency": "High"},
}

# Severity based on both damage type urgency AND bbox area ratio
URGENCY_WEIGHT = {"High": 2, "Medium": 1, "Low": 0}

def get_severity(box_area: int, img_area: int, damage_type: str):
    """
    Combines area ratio with the known urgency of the damage type.
    A small pothole can still be HIGH; a large stripe is still LOW.
    """
    ratio = box_area / img_area
    type_urgency = REPAIR_INFO.get(damage_type, {}).get("urgency", "Low")
    urgency_score = URGENCY_WEIGHT.get(type_urgency, 0)

    if ratio > 0.05 or urgency_score == 2:
        return "HIGH",   "#ef4444"
    if ratio > 0.02 or urgency_score == 1:
        return "MEDIUM", "#f59e0b"
    return "LOW", "#22c55e"

def severity_to_map_color(severity: str) -> str:
    return {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#22c55e"}.get(severity, "#6b7280")

# ── Database ─────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "road_damage.db")

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            scanned_at  TEXT    NOT NULL,
            location    TEXT,
            lat         REAL,
            lon         REAL,
            total       INTEGER,
            high        INTEGER,
            medium      INTEGER,
            low         INTEGER,
            image_path  TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id     INTEGER NOT NULL REFERENCES scans(id),
            type        TEXT,
            confidence  REAL,
            severity    TEXT,
            color       TEXT,
            bbox        TEXT,
            lat         REAL,
            lon         REAL
        )
    """)
    con.commit()
    con.close()

init_db()

# ── Image storage directory (replaces base64-in-SQLite) ──────────────────────
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "scan_images")
os.makedirs(IMAGES_DIR, exist_ok=True)

def get_db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

# ── Serve Frontend ──────────────────────────────────────────────────────────
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(frontend_dir, "index.html"))

# ── Helpers ──────────────────────────────────────────────────────────────────
def smart_fallback_chat(msg: str, damage_data: list) -> str:
    msg_lower = msg.lower()
    total  = len(damage_data)
    high   = [d for d in damage_data if d.get("severity") == "HIGH"]
    medium = [d for d in damage_data if d.get("severity") == "MEDIUM"]
    low    = [d for d in damage_data if d.get("severity") == "LOW"]
    type_counts = {}
    for d in damage_data:
        t = d.get("type", "Unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    if any(w in msg_lower for w in ["hello", "hi", "hey"]):
        if total == 0:
            return "👋 Hello! I'm your Road Damage Assistant. Upload an image to get started!"
        return (f"👋 Hello! Tracking {total} damage instance(s) — "
                f"{len(high)} high, {len(medium)} medium, {len(low)} low severity.")

    if total == 0 and not any(w in msg_lower for w in ["what", "how", "explain", "define"]):
        return "ℹ️ No scan data yet. Upload a road image with location to begin."

    if any(w in msg_lower for w in ["urgent", "critical", "danger", "emergency", "high"]):
        if not high:
            return "✅ No HIGH severity damage detected."
        types = ", ".join(set(d["type"] for d in high))
        return (f"🚨 {len(high)} HIGH severity issue(s): {types}.\n\n"
                "Immediate action required — dispatch a repair crew within 24–48 hours.")

    if "pothole" in msg_lower:
        ph = [d for d in damage_data if "pothole" in d.get("type", "").lower()]
        if not ph:
            return "✅ No potholes detected."
        avg_conf = round(sum(d.get("confidence", 0) for d in ph) / len(ph), 1)
        return (f"🕳️ {len(ph)} pothole(s) detected (avg confidence: {avg_conf}%).\n\n"
                "Repair: Hot-mix asphalt patching. Cost: $50–$500 per pothole.")

    if any(w in msg_lower for w in ["cost", "price", "budget", "estimate"]):
        if total == 0:
            return "No damage data to estimate costs. Please run a scan first."
        lines = [f"  • {count}× {dtype}: {REPAIR_INFO.get(dtype, {}).get('cost','N/A')} — {REPAIR_INFO.get(dtype, {}).get('method','N/A')}"
                 for dtype, count in type_counts.items()]
        return "💰 Repair Cost Estimates:\n\n" + "\n".join(lines) + "\n\n⚠️ Costs are indicative estimates."

    if any(w in msg_lower for w in ["repair", "fix", "how to", "solution", "method"]):
        if total == 0:
            return "No damage detected — no repairs needed."
        lines = [f"  • {dtype} → {REPAIR_INFO.get(dtype, {}).get('method','specialist assessment')} (urgency: {REPAIR_INFO.get(dtype, {}).get('urgency','N/A')})"
                 for dtype in type_counts]
        return "🔧 Recommended Repair Methods:\n\n" + "\n".join(lines)

    if any(w in msg_lower for w in ["summary", "overview", "total", "how many", "count"]):
        if total == 0:
            return "✅ Road is in good condition — no damage detected."
        type_str = ", ".join(f"{v}× {k}" for k, v in type_counts.items())
        return (f"📊 Scan Summary:\n\nTotal: {total}\n"
                f"🔴 High: {len(high)}  🟡 Medium: {len(medium)}  🟢 Low: {len(low)}\n\n"
                f"Types: {type_str}")

    if any(w in msg_lower for w in ["map", "location", "gps", "where"]):
        if total == 0:
            return "No damage locations yet. Run a scan first."
        locs = [f"  • {d['type']} ({d['severity']}) @ ({d.get('lat','N/A')}, {d.get('lon','N/A')})"
                for d in damage_data]
        return f"📍 {total} damage point(s):\n\n" + "\n".join(locs)

    if any(w in msg_lower for w in ["recommend", "priority", "next step", "action"]):
        if total == 0:
            return "✅ No damage. Continue routine inspections every 3–6 months."
        steps = []
        if high:
            steps.append(f"1. 🚨 IMMEDIATE: Repair {len(high)} high-severity issue(s) within 24–48 hrs.")
        if medium:
            steps.append(f"{len(steps)+1}. 📋 WITHIN 2 WEEKS: Address {len(medium)} medium-severity issue(s).")
        if low:
            steps.append(f"{len(steps)+1}. ✅ ROUTINE: Schedule {len(low)} low-severity repair(s) next cycle.")
        steps.append(f"{len(steps)+1}. 📸 Re-scan after repairs to confirm resolution.")
        return "📋 Action Plan:\n\n" + "\n".join(steps)

    if total > 0:
        return (f"Tracking {total} damage instance(s) — {len(high)} high, {len(medium)} medium, {len(low)} low. "
                "Ask about: costs, repairs, urgent issues, locations, or a summary.")
    return "I'm your Road Damage Assistant. Upload a road image with location to begin."


# ── Pydantic Models (no API key fields — key lives in .env) ──────────────────
class ChatRequest(BaseModel):
    message: str
    damage_data: Optional[List[dict]] = []

class ReportRequest(BaseModel):
    detections: List[dict]

# ── Endpoints ─────────────────────────────────────────────────────────────────

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB

@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    location_name: Optional[str] = Form(None)
):
    # ── Input validation ──────────────────────────────────────────────────────
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}. Use JPEG or PNG.")

    contents = await file.read()

    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Image too large. Maximum size is 10 MB.")

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Please upload a valid JPEG or PNG.")

    img_area = img.shape[0] * img.shape[1]
    results = model(img, conf=0.25)

    detections = []
    annotated = img.copy()

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_area = (x2 - x1) * (y2 - y1)
            label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"Class{cls_id}"

            # Severity now considers damage type, not just bbox size
            severity, color_hex = get_severity(box_area, img_area, label)

            bgr = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, 3)
            cv2.putText(annotated, f"{label} | {severity}", (x1, max(y1-10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)

            # Slightly jitter multiple detections around the provided GPS point
            d_lat = lat + random.uniform(-0.0003, 0.0003) if lat is not None else None
            d_lon = lon + random.uniform(-0.0003, 0.0003) if lon is not None else None

            detections.append({
                "type": label,
                "confidence": round(conf * 100, 1),
                "severity": severity,
                "color": color_hex,
                "bbox": [x1, y1, x2, y2],
                "lat": round(d_lat, 6) if d_lat is not None else None,
                "lon": round(d_lon, 6) if d_lon is not None else None,
            })

    _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    annotated_b64 = base64.b64encode(buffer).decode('utf-8')

    _, orig_buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    original_b64 = base64.b64encode(orig_buf).decode('utf-8')

    summary = {
        "total":  len(detections),
        "high":   len([d for d in detections if d["severity"] == "HIGH"]),
        "medium": len([d for d in detections if d["severity"] == "MEDIUM"]),
        "low":    len([d for d in detections if d["severity"] == "LOW"]),
    }

    # ── Save annotated image to disk instead of base64 in DB ─────────────────
    scan_ts = datetime.now()
    image_filename = f"scan_{scan_ts.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    image_path = os.path.join(IMAGES_DIR, image_filename)
    with open(image_path, 'wb') as f:
        f.write(buffer)

    # ── Save to DB ────────────────────────────────────────────────────────────
    con = get_db()
    cur = con.cursor()
    cur.execute("""
        INSERT INTO scans (scanned_at, location, lat, lon, total, high, medium, low, image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (scan_ts.isoformat(), location_name, lat, lon,
          summary["total"], summary["high"], summary["medium"], summary["low"],
          image_path))
    scan_id = cur.lastrowid
    for d in detections:
        cur.execute("""
            INSERT INTO detections (scan_id, type, confidence, severity, color, bbox, lat, lon)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (scan_id, d["type"], d["confidence"], d["severity"],
              d["color"], json.dumps(d["bbox"]), d.get("lat"), d.get("lon")))
    con.commit()
    con.close()

    return {
        "scan_id": scan_id,
        "detections": detections,
        "annotated_image": annotated_b64,
        "original_image": original_b64,
        "summary": summary,
        "location": {"lat": lat, "lon": lon, "name": location_name}
    }


@app.get("/scans")
def get_all_scans():
    """Return all scans for history view."""
    con = get_db()
    rows = con.execute(
        "SELECT id, scanned_at, location, lat, lon, total, high, medium, low FROM scans ORDER BY scanned_at DESC"
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


@app.get("/scans/{scan_id}")
def get_scan(scan_id: int):
    """Return a single scan with all its detections."""
    con = get_db()
    scan = con.execute("SELECT * FROM scans WHERE id=?", (scan_id,)).fetchone()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    dets = con.execute("SELECT * FROM detections WHERE scan_id=?", (scan_id,)).fetchall()
    con.close()
    result = dict(scan)
    result["detections"] = [dict(d) for d in dets]
    return result


@app.get("/map-data")
def get_map_data():
    """Returns all detections with GPS coords for the live map."""
    con = get_db()
    rows = con.execute("""
        SELECT d.id, d.type, d.confidence, d.severity, d.color, d.lat, d.lon,
               s.scanned_at, s.location, s.id as scan_id
        FROM detections d
        JOIN scans s ON d.scan_id = s.id
        WHERE d.lat IS NOT NULL AND d.lon IS NOT NULL
        ORDER BY s.scanned_at DESC
    """).fetchall()
    con.close()

    points = []
    for r in rows:
        r = dict(r)
        r["map_color"] = severity_to_map_color(r["severity"])
        r["radius"] = {"HIGH": 14, "MEDIUM": 10, "LOW": 7}.get(r["severity"], 8)
        points.append(r)
    return {"points": points}


@app.delete("/scans/{scan_id}")
def delete_scan(scan_id: int):
    con = get_db()
    # Also remove the image file from disk if it exists
    scan = con.execute("SELECT image_path FROM scans WHERE id=?", (scan_id,)).fetchone()
    if scan and scan["image_path"] and os.path.exists(scan["image_path"]):
        os.remove(scan["image_path"])
    con.execute("DELETE FROM detections WHERE scan_id=?", (scan_id,))
    con.execute("DELETE FROM scans WHERE id=?", (scan_id,))
    con.commit()
    con.close()
    return {"deleted": scan_id}


@app.post("/report")
def generate_report(req: ReportRequest):
    detections = req.detections
    if not detections:
        return {"report": "✅ No damage detected. Road is in good condition."}

    # Uses GROQ_API_KEY from .env — not from the request
    if GROQ_API_KEY:
        summary_str = ", ".join([f"{d['type']} ({d['severity']})" for d in detections])
        prompt = (f"You are a municipal road inspection AI. Generate a professional road damage report for: {summary_str}. "
                  "Write 3 paragraphs: (1) damage summary, (2) risk assessment, (3) repair recommendations.")
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={"model": "llama3-8b-8192", "messages": [{"role": "user", "content": prompt}], "max_tokens": 400},
                timeout=15
            )
            resp.raise_for_status()
            return {"report": resp.json()["choices"][0]["message"]["content"]}
        except Exception as e:
            print(f"Groq API error: {e}")

    high   = [d for d in detections if d["severity"] == "HIGH"]
    medium = [d for d in detections if d["severity"] == "MEDIUM"]
    low    = [d for d in detections if d["severity"] == "LOW"]
    type_counts = {}
    for d in detections:
        type_counts[d.get("type", "Unknown")] = type_counts.get(d.get("type", "Unknown"), 0) + 1

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    div = "=" * 50
    report = f"""ROAD DAMAGE INSPECTION REPORT
{div}
Date/Time      : {now}
Total Detections: {len(detections)}
🔴 High   : {len(high)}
🟡 Medium : {len(medium)}
🟢 Low    : {len(low)}

{div}
DAMAGE BREAKDOWN
{div}
"""
    for dtype, count in type_counts.items():
        info = REPAIR_INFO.get(dtype, {"cost": "N/A", "method": "specialist assessment", "urgency": "N/A"})
        icon = "🔴" if info["urgency"] == "High" else "🟡" if info["urgency"] == "Medium" else "🟢"
        report += (f"  {icon} {dtype} × {count}\n"
                   f"     Repair  : {info['method']}\n"
                   f"     Est Cost: {info['cost']}\n"
                   f"     Urgency : {info['urgency']}\n\n")

    report += f"{div}\nDETECTION DETAILS\n{div}\n"
    for d in detections:
        icon = "🔴" if d["severity"] == "HIGH" else "🟡" if d["severity"] == "MEDIUM" else "🟢"
        gps = f"({d.get('lat','N/A')}, {d.get('lon','N/A')})" if d.get("lat") else "no GPS"
        report += f"  {icon} {d['type']} — {d['severity']} (Conf: {d['confidence']}%, {gps})\n"

    report += f"\n{div}\nRECOMMENDATIONS\n{div}\n"
    if high:
        report += f"🚨 IMMEDIATE: {len(high)} high-severity issue(s) — deploy crew within 24–48 hrs.\n"
    if medium:
        report += f"📋 WITHIN 2 WEEKS: {len(medium)} medium-severity issue(s) need prompt attention.\n"
    if low:
        report += f"✅ ROUTINE: {len(low)} low-severity issue(s) — include in next maintenance cycle.\n"
    report += f"\n{div}\nGenerated by Road Damage Detection System v2.0\n"
    return {"report": report}


@app.post("/chat")
def chat(req: ChatRequest):
    damage_data = req.damage_data or []
    # Uses GROQ_API_KEY from .env — not from the request
    if GROQ_API_KEY:
        context = f"Road damage data: {json.dumps(damage_data)}" if damage_data else "No scan data yet."
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "llama3-8b-8192",
                    "messages": [
                        {"role": "system", "content": f"You are a municipal road management AI. {context}"},
                        {"role": "user", "content": req.message}
                    ],
                    "max_tokens": 300
                },
                timeout=15
            )
            resp.raise_for_status()
            return {"response": resp.json()["choices"][0]["message"]["content"]}
        except Exception as e:
            print(f"Groq API error: {e}")
    return {"response": smart_fallback_chat(req.message, damage_data)}


@app.get("/health")
def health():
    groq_configured = bool(GROQ_API_KEY)
    return {
        "status": "ok",
        "model": "YOLOv8n",
        "classes": len(CLASS_NAMES),
        "db": DB_PATH,
        "groq_configured": groq_configured
    }
