import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import cv2
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database import SessionLocal
from src.dependencies import get_db
from src.models import Vehicle
from src.schemas import VehicleOut
from src.utils import det_model, detect_plate, recognize_plate

router = APIRouter(tags=["Dashboard"])

AGGREGATION_WINDOW = timedelta(seconds=30)
SAVE_COOLDOWN = timedelta(seconds=30)
JPEG_QUALITY = 80
FPS = 30


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _should_save(plate: str, cache: dict[str, datetime], now: datetime) -> bool:
    ts = cache.get(plate)
    return ts is None or now - ts >= SAVE_COOLDOWN


def _encode_frame(frame) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError
    return buf.tobytes()


def gen_frames():
    cap = cv2.VideoCapture(0)
    try:
        with SessionLocal() as session:
            plate_counts: defaultdict[str, int] = defaultdict(int)
            last_saved: dict[str, datetime] = {}
            prev_plate: str | None = None
            window_start = _utcnow()
            while True:
                success, frame = cap.read()
                if not success:
                    continue
                now = _utcnow()
                out_frame = frame.copy()
                detections = detect_plate(frame)
                for x1, y1, x2, y2, conf, cls in detections:
                    cv2.rectangle(out_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(out_frame, f"{det_model.names[cls]} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue
                    plate = recognize_plate(roi)
                    if plate:
                        plate_counts[plate] += 1
                        cv2.putText(out_frame, plate, (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                if now - window_start >= AGGREGATION_WINDOW and plate_counts:
                    best_plate, _ = max(plate_counts.items(), key=lambda kv: kv[1])
                    if best_plate != prev_plate and _should_save(best_plate, last_saved, now):
                        session.add(Vehicle(plate_number=best_plate, entry_time=now))
                        session.commit()
                        last_saved[best_plate] = now
                        prev_plate = best_plate
                    plate_counts.clear()
                    window_start = now
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + _encode_frame(out_frame) + b"\r\n")
                time.sleep(1 / FPS)
    finally:
        cap.release()


@router.get("/stream-feed")
async def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.get("/vehicles", response_model=list[VehicleOut])
async def get_vehicles(db: Session = Depends(get_db)):
    stmt = select(Vehicle).order_by(Vehicle.entry_time.desc()).limit(10)
    vehicles = db.scalars(stmt).all()
    return [VehicleOut(id=v.id, plate_number=v.plate_number, datetime=v.entry_time) for v in vehicles]
