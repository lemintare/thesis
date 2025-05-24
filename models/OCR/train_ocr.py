"""Полный скрипт обучения OCR-модели EfficientNet-V2 + BiLSTM + CTC

Структура датасета:
    dataset/
        train/img/*.jpg (имя файла = метка)
        val/img/*.jpg
        test/img/*.jpg

Как запустить:
    uv run python train_ocr.py --data_dir /path/to/dataset \
                        --epochs 50 --batch_size 16 --lr 1e-4
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import timm

try:
    from clearml import Task
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False

# --------------- Globals --------------- #
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg'}
VOCABULARY = "-1234567890ABEKMHOPCTYX"
char_to_idx = {c: i + 1 for i, c in enumerate(VOCABULARY)}
char_to_idx['blank'] = 0
idx_to_char = {i: c for c, i in char_to_idx.items()}


def natural_key(s: str):
    """Sort helper: '2.jpg' < '10.jpg'."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


# --------------- Dataset --------------- #
class PlateDataset(Dataset):
    """Читает изображения и метку (из имени файла)"""

    def __init__(self, dir_path: str | Path, transform=None):
        self.root = Path(dir_path)
        img_folder = self.root / 'img'
        if img_folder.is_dir():
            self.root = img_folder
        self.transform = transform

        self.image_files: List[Path] = sorted(
            [p for p in self.root.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS],
            key=lambda p: natural_key(p.name)
        )
        if not self.image_files:
            raise RuntimeError(f"Не найдено изображений в {self.root}")

    # --- utils
    @staticmethod
    def label_to_indices(label: str) -> List[int]:
        indices = []
        for ch in label:
            if ch not in char_to_idx:
                raise ValueError(f"Unknown symbol '{ch}' in label '{label}'")
            indices.append(char_to_idx[ch])
        return indices

    # --- Dataset API
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)['image']
        label_str = img_path.stem
        indices = self.label_to_indices(label_str)
        return image, torch.tensor(indices, dtype=torch.long), label_str


# --------------- Collate fn --------------- #
def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, str]]):
    imgs, label_tensors, label_strs = zip(*batch)
    imgs = torch.stack(imgs, 0)
    targets = torch.cat(label_tensors)
    target_lengths = torch.tensor([len(t) for t in label_tensors], dtype=torch.long)
    return imgs, targets, target_lengths, label_strs


# --------------- Metrics --------------- #
class PlatesRecognized:
    def __init__(self):
        self.reset()

    def update_state(self, y_true: List[str], y_pred: List[str]):
        for t, p in zip(y_true, y_pred):
            if t == p:
                self.correct += 1
            self.total += 1

    def result(self):
        return self.correct / self.total if self.total else 0.0

    def reset(self):
        self.correct = 0
        self.total = 0


class SymbolsRecognized:
    def __init__(self):
        self.reset()

    def update_state(self, y_true: List[str], y_pred: List[str]):
        for t, p in zip(y_true, y_pred):
            l = min(len(t), len(p))
            self.correct += sum(t[i] == p[i] for i in range(l))
            self.total += len(t)

    def result(self):
        return self.correct / self.total if self.total else 0.0

    def reset(self):
        self.correct = 0
        self.total = 0


# --------------- Model --------------- #
class EfficientNetV2LOCR(nn.Module):
    def __init__(self, num_chars: int, image_shape: Tuple[int, int, int]):
        super().__init__()
        self.backbone = timm.create_model('efficientnetv2_l', pretrained=False, num_classes=0, features_only=True)
        # пробный прогон, чтобы узнать число каналов
        with torch.no_grad():
            dummy = torch.zeros(1, *image_shape)
            C = self.backbone(dummy)[-1].shape[1]
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(input_size=C, hidden_size=256, num_layers=1,
                            batch_first=True, bidirectional=True, dropout=0.3)
        self.bn = nn.BatchNorm1d(256)
        self.fc = nn.Linear(256, num_chars)

    def forward(self, x):
        feat = self.backbone(x)[-1]  # (B, C, H, W)
        B, C, H, W = feat.shape
        x = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, S, C)
        x = self.dropout(x)
        lstm_out, _ = self.lstm(x)
        fwd, bwd = lstm_out[..., :256], lstm_out[..., 256:]
        x = (fwd + bwd) / 2  # (B, S, 256)
        x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=2)
        return x.permute(1, 0, 2)  # (S, B, C)


# --------------- Helpers --------------- #
@torch.no_grad()
def greedy_decode(output: torch.Tensor) -> List[str]:
    """CTC greedy decoding. Output shape: (S, B, C)"""
    B = output.size(1)
    decoded: List[str] = []
    for b in range(B):
        probs = output[:, b]  # (S, C)
        best = probs.argmax(dim=1)
        best = torch.unique_consecutive(best)
        pred = ''.join(idx_to_char[i.item()] for i in best if i.item() != char_to_idx['blank'])
        decoded.append(pred)
    return decoded


# --------------- Training --------------- #
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    plates_metric = PlatesRecognized()
    symbols_metric = SymbolsRecognized()

    for imgs, tgt_flat, tgt_lens, tgt_strs in loader:
        imgs, tgt_flat, tgt_lens = imgs.to(device), tgt_flat.to(device), tgt_lens.to(device)
        optimizer.zero_grad()
        out = model(imgs)  # (S, B, C)
        S = out.size(0)
        B = out.size(1)
        input_lens = torch.full((B,), S, dtype=torch.long, device=device)
        loss = criterion(out, tgt_flat, input_lens, tgt_lens)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds = greedy_decode(out)
        plates_metric.update_state(tgt_strs, preds)
        symbols_metric.update_state(tgt_strs, preds)

    n_batches = len(loader)
    return epoch_loss / n_batches, plates_metric.result(), symbols_metric.result()


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    plates_metric = PlatesRecognized()
    symbols_metric = SymbolsRecognized()

    for imgs, tgt_flat, tgt_lens, tgt_strs in loader:
        imgs, tgt_flat, tgt_lens = imgs.to(device), tgt_flat.to(device), tgt_lens.to(device)
        out = model(imgs)
        S = out.size(0)
        B = out.size(1)
        input_lens = torch.full((B,), S, dtype=torch.long, device=device)
        loss = criterion(out, tgt_flat, input_lens, tgt_lens)
        val_loss += loss.item()
        preds = greedy_decode(out)
        plates_metric.update_state(tgt_strs, preds)
        symbols_metric.update_state(tgt_strs, preds)

    n_batches = len(loader)
    return val_loss / n_batches, plates_metric.result(), symbols_metric.result()


# --------------- Main --------------- #

def main():
    parser = argparse.ArgumentParser(description="Train plate OCR model")
    parser.add_argument('--data_dir', type=str, required=True, help='dataset root containing train/val/test')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--out', type=str, default='best.pt', help='file to save best weights')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ClearML task (optional)
    if CLEARML_AVAILABLE:
        task = Task.init(project_name='thesis', task_name='OCR_EfficientNetV2L', output_uri=True)
        task.connect(args)
    else:
        task = None

    # Transforms
    transform = A.Compose([
        A.Resize(width=200, height=100),
        A.ColorJitter(0.2, 0.2, 0.2, 0.2, p=0.5),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])

    # Datasets & loaders
    data_dir = Path(args.data_dir)
    train_ds = PlateDataset(data_dir / 'train', transform)
    val_ds   = PlateDataset(data_dir / 'val', transform)

    loader_kwargs = dict(batch_size=args.batch_size, pin_memory=True,
                         num_workers=args.workers, collate_fn=collate_fn)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # Model, loss, optim
    model = EfficientNetV2LOCR(num_chars=len(char_to_idx), image_shape=(3, 100, 200)).to(device)
    criterion = nn.CTCLoss(blank=char_to_idx['blank'], zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_plate_acc, train_sym_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_plate_acc, val_sym_acc = validate(
            model, val_loader, criterion, device)

        # Logging
        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train loss {train_loss:.4f} | Train plate acc {train_plate_acc:.4f} | Train sym acc {train_sym_acc:.4f}\n"
              f"                        "
              f"Val loss {val_loss:.4f} | Val plate acc {val_plate_acc:.4f} | Val sym acc {val_sym_acc:.4f}")

        if task:
            task.logger.report_scalar("Loss", "train", train_loss, epoch)
            task.logger.report_scalar("Loss", "val", val_loss, epoch)
            task.logger.report_scalar("PlateAcc", "train", train_plate_acc, epoch)
            task.logger.report_scalar("PlateAcc", "val", val_plate_acc, epoch)
            task.logger.report_scalar("SymAcc", "train", train_sym_acc, epoch)
            task.logger.report_scalar("SymAcc", "val", val_sym_acc, epoch)

        if val_plate_acc > best_acc:
            best_acc = val_plate_acc
            torch.save(model.state_dict(), args.out)
            print(f"\n✔ Saved new best model to {args.out} (val plate acc {val_plate_acc:.4f})\n")


if __name__ == "__main__":
    main()
