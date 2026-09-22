from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2

from config import CLASS_NAMES
from modules.math_model import bbox_area_xyxy


def _has_boxes(result: Any) -> bool:
    """Cek apakah hasil YOLO memiliki bounding box."""
    try:
        return result.boxes is not None and len(result.boxes) > 0
    except Exception:
        return False


def _read_image(image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("Gambar gagal dibaca OpenCV. Pastikan file gambar tidak rusak.")
    return image


def _draw_label(image, x1: int, y1: int, label: str) -> None:
    text_color = (35, 42, 55)
    panel_color = (245, 247, 250)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    y_top = max(0, y1 - th - 12)
    cv2.rectangle(image, (x1, y_top), (x1 + tw + 12, y1), panel_color, -1)
    cv2.putText(
        image,
        label,
        (x1 + 6, max(15, y1 - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        text_color,
        1,
        cv2.LINE_AA,
    )


def _extract_detections(result: Any, image_area: float) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    if not _has_boxes(result):
        return detections

    for box in result.boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(float)
        x1, y1, x2, y2 = xyxy.tolist()
        cls_id = int(box.cls[0].cpu().item())
        confidence = float(box.conf[0].cpu().item())

        class_name = result.names.get(
            cls_id,
            CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id),
        )

        area_px = bbox_area_xyxy((x1, y1, x2, y2))
        area_pct = (area_px / image_area) * 100 if image_area else 0

        detections.append({
            "class_id": cls_id,
            "class_name": class_name,
            "confidence": round(confidence, 4),
            "x1": round(x1, 2),
            "y1": round(y1, 2),
            "x2": round(x2, 2),
            "y2": round(y2, 2),
            "area_px": round(area_px, 2),
            "area_pct": round(area_pct, 4),
        })

    detections.sort(key=lambda item: item.get("confidence", 0), reverse=True)
    return detections


def _draw_detections(image, detections: List[Dict[str, Any]]) -> None:
    box_color = (88, 103, 130)
    h, w = image.shape[:2]

    for item in detections:
        x1 = max(0, min(w - 1, int(item["x1"])))
        y1 = max(0, min(h - 1, int(item["y1"])))
        x2 = max(0, min(w - 1, int(item["x2"])))
        y2 = max(0, min(h - 1, int(item["y2"])))

        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)
        label = f"{item['class_name']} {item['confidence'] * 100:.2f}%"
        _draw_label(image, x1, y1, label)


def _draw_no_detection_note(image) -> None:
    """Tulis catatan kecil pada gambar saat tidak ada deteksi."""
    note = "Tidak ada objek terdeteksi"
    text_color = (35, 42, 55)
    panel_color = (245, 247, 250)
    cv2.rectangle(image, (18, 18), (335, 58), panel_color, -1)
    cv2.putText(image, note, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)


def predict_image(
    model_path: Path,
    image_path: Path,
    output_path: Path,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    fallback_conf: float = 0.01,
) -> Dict[str, Any]:
    """Inferensi citra YOLOv8 dengan fallback confidence rendah.

    Alur baru:
    1. Prediksi memakai confidence dari form.
    2. Jika tidak ada deteksi, ulangi sekali memakai fallback_conf 0.01.
    3. Jika masih kosong, sistem menampilkan status tidak terdeteksi secara jelas.

    Catatan:
    YOLO object detection hanya memberi confidence jika ada bounding box.
    Karena itu, fallback ini membantu melihat kandidat lemah, bukan memaksa hasil agar selalu benar.
    """
    from ultralytics import YOLO

    model_path = Path(model_path)
    image_path = Path(image_path)
    output_path = Path(output_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {model_path}. "
            "Pastikan file best.pt ada di runs/malaria_yolov8/weights/best.pt atau di folder utama project."
        )
    if not image_path.exists():
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

    conf = max(0.01, min(1.0, float(conf)))
    fallback_conf = max(0.01, min(conf, float(fallback_conf)))
    iou = max(0.01, min(1.0, float(iou)))
    imgsz = int(imgsz)

    model = YOLO(str(model_path))

    primary_results = model.predict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        save=False,
        verbose=False,
    )
    result = primary_results[0]
    used_fallback = False
    used_conf = conf

    if not _has_boxes(result) and fallback_conf < conf:
        fallback_results = model.predict(
            source=str(image_path),
            conf=fallback_conf,
            iou=iou,
            imgsz=imgsz,
            save=False,
            verbose=False,
        )
        result = fallback_results[0]
        used_fallback = _has_boxes(result)
        used_conf = fallback_conf if used_fallback else conf

    image = _read_image(image_path)
    h, w = image.shape[:2]
    image_area = float(w * h)

    detections = _extract_detections(result, image_area)

    if detections:
        _draw_detections(image, detections)
    else:
        _draw_no_detection_note(image)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)

    return {
        "output_path": str(output_path),
        "image_width": w,
        "image_height": h,
        "detections": detections,
        "count": len(detections),
        "requested_conf": conf,
        "used_conf": used_conf,
        "fallback_conf": fallback_conf,
        "used_fallback": used_fallback,
        "model_path": str(model_path),
    }
