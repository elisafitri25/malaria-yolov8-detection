"""Pemodelan matematis pendukung YOLOv8 malaria.

Modul ini sengaja dibuat eksplisit agar skripsi Bab 3 dapat menjelaskan
normalisasi koordinat YOLO, transformasi bounding box, IoU, confidence,
dan komponen loss secara ringkas namun ilmiah.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import math


@dataclass(frozen=True)
class YoloBox:
    """Representasi bounding box format YOLO.

    Semua nilai xc, yc, w, h berada pada rentang [0, 1].
    xc, yc = titik tengah bbox ternormalisasi.
    w, h     = lebar dan tinggi bbox ternormalisasi.
    """

    cls: int
    xc: float
    yc: float
    w: float
    h: float


def sigmoid(z: float) -> float:
    """Fungsi aktivasi logistik untuk mengubah logit menjadi probabilitas."""
    return 1.0 / (1.0 + math.exp(-z))


def yolo_to_xyxy(box: YoloBox, image_w: int, image_h: int) -> Tuple[int, int, int, int]:
    """Konversi YOLO normalized xywh menjadi pixel xyxy.

    Rumus:
        x1 = (xc - w/2) * W
        y1 = (yc - h/2) * H
        x2 = (xc + w/2) * W
        y2 = (yc + h/2) * H
    """
    x1 = int(round((box.xc - box.w / 2) * image_w))
    y1 = int(round((box.yc - box.h / 2) * image_h))
    x2 = int(round((box.xc + box.w / 2) * image_w))
    y2 = int(round((box.yc + box.h / 2) * image_h))
    return max(0, x1), max(0, y1), min(image_w - 1, x2), min(image_h - 1, y2)


def xyxy_to_yolo(cls: int, x1: float, y1: float, x2: float, y2: float, image_w: int, image_h: int) -> YoloBox:
    """Konversi pixel xyxy menjadi YOLO normalized xywh."""
    xc = ((x1 + x2) / 2) / image_w
    yc = ((y1 + y2) / 2) / image_h
    w = abs(x2 - x1) / image_w
    h = abs(y2 - y1) / image_h
    return YoloBox(cls=cls, xc=xc, yc=yc, w=w, h=h)


def bbox_area_xyxy(box: Tuple[float, float, float, float]) -> float:
    """Luas bbox dalam piksel persegi."""
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """Intersection over Union.

    IoU = Area(A ∩ B) / Area(A ∪ B)
    Nilai mendekati 1 berarti prediksi sangat dekat dengan ground truth.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    inter = bbox_area_xyxy((ix1, iy1, ix2, iy2))
    union = bbox_area_xyxy(a) + bbox_area_xyxy(b) - inter
    return 0.0 if union <= 0 else inter / union


def detection_confidence(objectness: float, class_probability: float) -> float:
    """Confidence deteksi YOLO secara konseptual.

    confidence = P(object) × P(class | object)
    """
    return objectness * class_probability


def ciou_components(pred: Tuple[float, float, float, float], gt: Tuple[float, float, float, float]) -> dict:
    """Komponen ringkas CIoU untuk penjelasan matematis.

    Implementasi training sebenarnya berada di Ultralytics. Fungsi ini untuk
    dokumentasi dan analisis konsep: IoU, jarak pusat, diagonal enclosing box.
    """
    px1, py1, px2, py2 = pred
    gx1, gy1, gx2, gy2 = gt
    iou = iou_xyxy(pred, gt)

    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
    center_dist_sq = (pcx - gcx) ** 2 + (pcy - gcy) ** 2

    cx1, cy1 = min(px1, gx1), min(py1, gy1)
    cx2, cy2 = max(px2, gx2), max(py2, gy2)
    diag_sq = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2

    return {
        "iou": iou,
        "center_distance_squared": center_dist_sq,
        "enclosing_diagonal_squared": diag_sq,
        "distance_penalty": 0.0 if diag_sq <= 0 else center_dist_sq / diag_sq,
    }
