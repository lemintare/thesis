from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from typing import Optional
from src.config import settings
import cv2
import time
import numpy as np
from src.database import SessionLocal
import os 

from ultralytics import YOLO
import onnxruntime as ort


pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')

SECRET_KEY = settings.SECRET_KEY
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES
ALGORITHM = settings.ALGORITHM

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta if expires_delta else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({'exp': expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt

def decode_access_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return {}
    

# ————— Model inference —————

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yolo.pt')

det_model = YOLO(MODEL_PATH)

_BASE_DIR = os.path.dirname(__file__)  
_MODEL_DIR = os.path.join(_BASE_DIR, "models")

_OCR_PATH = os.path.join(_MODEL_DIR, "efficientnetv2locr.onnx")

ocr_sess = ort.InferenceSession(_OCR_PATH, providers=["CPUExecutionProvider"])

ocr_input_name  = ocr_sess.get_inputs()[0].name
ocr_output_name = ocr_sess.get_outputs()[0].name

vocabulary = "-1234567890ABEKMHOPCTYX"
char_to_idx = {char: idx + 1 for idx, char in enumerate(vocabulary)}
char_to_idx['blank'] = 0
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

def decode_plate_tensor(onnx_out: np.ndarray) -> str:
    """
    Greedy CTC: onnx_out shape = (seq_len, batch=1, num_classes)
    1) squeeze по batch
    2) argmax по классам
    3) collapse repeats + drop blank (0)
    """
    if onnx_out.ndim == 3 and onnx_out.shape[0] == 1:
        seq = onnx_out[0]
    else:
        seq = onnx_out.squeeze(1) 

    pred_ids = np.argmax(seq, axis=1).tolist()
    chars = []
    prev = None
    for idx in pred_ids:
        if idx != prev and idx != char_to_idx['blank']:
            chars.append(idx_to_char.get(idx, ""))
        prev = idx
    return "".join(chars)

def recognize_plate(roi: np.ndarray) -> str:
    """
    Прогоняет обрезанный ROI (BGR) через OCR-модель.
    Возвращает строку распознанного номера.
    """
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (200, 100)).astype(np.float32) / 255.0
    inp = img.transpose(2,0,1)[None, ...]
    out = ocr_sess.run([ocr_output_name], {ocr_input_name: inp})[0]
    return decode_plate_tensor(out)

def detect_plate(frame) -> list[tuple[int, int, int, int, float, int]]:
    """
    Возвращает список детекций номера:
    [(x1, y1, x2, y2, confidence, class_id), ...]
    """
    result = det_model(frame)[0]
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        cls  = int(box.cls[0].item())
        detections.append((x1, y1, x2, y2, conf, cls))
    return detections