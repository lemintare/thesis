import time
from datetime import datetime, timedelta
from collections import defaultdict

import cv2
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.utils import detect_plate, recognize_plate, det_model
from src.models import Vehicle
from src.database import SessionLocal
from src.dependencies import get_db
from src.schemas import VehicleOut

router = APIRouter(tags=['Dashboard'])

SAVE_COOLDOWN = timedelta(seconds=30)
_last_saved: dict[str, datetime] = {}

AGGREGATION_WINDOW = timedelta(seconds=30)

def gen_frames():
    cap = cv2.VideoCapture(0)
    session = SessionLocal()

    plate_counts: defaultdict[str, int] = defaultdict(int)
    window_start: datetime = datetime.utcnow()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            now = datetime.utcnow()
            out_frame = frame.copy()

            detections = detect_plate(frame)

            for x1, y1, x2, y2, conf, cls in detections:
                label = f"{det_model.names[cls]} {conf:.2f}"
                cv2.rectangle(out_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(out_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                roi = frame[y1:y2, x1:x2]
                plate = recognize_plate(roi)

                if plate:
                    plate_counts[plate] += 1
                    cv2.putText(out_frame, plate, (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            if now - window_start >= AGGREGATION_WINDOW and plate_counts:
                best_plate, _ = max(plate_counts.items(), key=lambda kv: kv[1])

                last = _last_saved.get(best_plate)
                if last is None or (now - last) > SAVE_COOLDOWN:
                    session.add(Vehicle(
                        plate_number=best_plate,
                        entry_time=now
                    ))
                    session.commit()
                    _last_saved[best_plate] = now
                    print(f"Saved to DB: {best_plate}")

                plate_counts.clear()
                window_start = now

            ret, buf = cv2.imencode(".jpg", out_frame)
            time.sleep(0.03) 
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
    finally:
        session.close()
        cap.release()


@router.get("/stream-feed")
def video_feed():
    """ MJPEG-стрим с рабочей логикой агрегации номеров. """
    return StreamingResponse(
        gen_frames(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

@router.get("/vehicles", response_model=list[VehicleOut])
def get_vehicles(db: Session = Depends(get_db)):
    """
    Вернуть 10 последних записей, отсортированных
    по времени въезда (entry_time) в порядке убывания.
    """
    stmt = (
        select(Vehicle)
        .order_by(Vehicle.entry_time.desc())
        .limit(10)
    )

    rows = db.execute(stmt).scalars().all()

    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No vehicles found"
        )

    return [
        VehicleOut(
            id=v.id,
            plate_number=v.plate_number,
            datetime=v.entry_time
        )
        for v in rows
    ]