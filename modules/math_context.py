from __future__ import annotations

from typing import Any, Dict

from config import (
    CLASS_NAMES,
    TARGET_PER_CLASS,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    INFECTED_DIR,
)
from modules.dataset_manager import count_dataset


DISPLAY_ORDER = ["Vivax", "Falciparum", "Malariae", "Ovale", "Knowlesi"]

FALLBACK_COUNTS = {
    "Vivax": 1425,
    "Falciparum": 918,
    "Malariae": 436,
    "Ovale": 312,
    "Knowlesi": 184,
}


def _format_number(value: float | int, digits: int = 3) -> str:
    """Format angka untuk tampilan Indonesia."""
    if isinstance(value, int):
        return f"{value:,}".replace(",", ".")
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return text.replace(".", ",")


def _get_source_counts() -> Dict[str, int]:
    """Ambil jumlah gambar dari folder data asli.

    Jika folder belum tersedia di komputer pengguna, halaman rumus tetap tampil
    memakai angka contoh penelitian.
    """
    try:
        counts = count_dataset(INFECTED_DIR)
        total = int(counts.get("TOTAL", {}).get("images", 0))
        if total > 0:
            return {cls: int(counts.get(cls, {}).get("images", 0)) for cls in CLASS_NAMES}
    except Exception:
        pass

    return FALLBACK_COUNTS.copy()


def build_math_context() -> Dict[str, Any]:
    counts = _get_source_counts()
    ordered_counts = {
        cls: int(counts.get(cls, 0))
        for cls in DISPLAY_ORDER
        if cls in counts
    }

    total_awal = sum(ordered_counts.values())
    jumlah_kelas = len(ordered_counts)
    target_per_class = int(TARGET_PER_CLASS)
    target_total = jumlah_kelas * target_per_class

    class_rows = []
    total_augmentasi = 0
    total_pengurangan = 0

    for cls, awal in ordered_counts.items():
        if awal < target_per_class:
            augmentasi = target_per_class - awal
            pengurangan = 0
            aksi = f"Tambah {augmentasi} citra"
            status = "Augmentasi"
        elif awal > target_per_class:
            augmentasi = 0
            pengurangan = awal - target_per_class
            aksi = f"Pilih {target_per_class} dari {awal} citra"
            status = "Pengurangan acak"
        else:
            augmentasi = 0
            pengurangan = 0
            aksi = "Sudah sesuai target"
            status = "Tetap"

        total_augmentasi += augmentasi
        total_pengurangan += pengurangan

        class_rows.append({
            "class_name": cls,
            "awal": awal,
            "target": target_per_class,
            "augmentasi": augmentasi,
            "pengurangan": pengurangan,
            "akhir": target_per_class,
            "aksi": aksi,
            "status": status,
        })

    train_per_class = int(target_per_class * TRAIN_RATIO)
    val_per_class = int(target_per_class * VAL_RATIO)
    test_per_class = target_per_class - train_per_class - val_per_class

    train_total = train_per_class * jumlah_kelas
    val_total = val_per_class * jumlah_kelas
    test_total = test_per_class * jumlah_kelas

    pixel_awal = 128
    pixel_normal = pixel_awal / 255

    image_w = 640
    image_h = 640
    x_center = 0.50
    y_center = 0.50
    box_w = 0.25
    box_h = 0.30

    x1 = int(round((x_center - box_w / 2) * image_w))
    y1 = int(round((y_center - box_h / 2) * image_h))
    x2 = int(round((x_center + box_w / 2) * image_w))
    y2 = int(round((y_center + box_h / 2) * image_h))

    box_width_px = x2 - x1
    box_height_px = y2 - y1
    box_area_px = box_width_px * box_height_px
    image_area_px = image_w * image_h
    box_area_percent = (box_area_px / image_area_px) * 100

    objectness = 0.90
    class_probability = 0.80
    confidence = objectness * class_probability

    area_prediksi = 80
    area_label = 90
    area_irisan = 50
    area_gabungan = area_prediksi + area_label - area_irisan
    iou = area_irisan / area_gabungan

    tp = 90
    fp = 10
    fn = 15
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)

    ap_values = [0.91, 0.88, 0.86, 0.84, 0.90]
    map_value = sum(ap_values) / len(ap_values)

    return {
        "display_order": DISPLAY_ORDER,
        "counts": ordered_counts,
        "class_rows": class_rows,
        "total_awal": total_awal,
        "jumlah_kelas": jumlah_kelas,
        "target_per_class": target_per_class,
        "target_total": target_total,
        "total_augmentasi": total_augmentasi,
        "total_pengurangan": total_pengurangan,
        "train_ratio": int(TRAIN_RATIO * 100),
        "val_ratio": int(VAL_RATIO * 100),
        "test_ratio": int(TEST_RATIO * 100),
        "train_per_class": train_per_class,
        "val_per_class": val_per_class,
        "test_per_class": test_per_class,
        "train_total": train_total,
        "val_total": val_total,
        "test_total": test_total,
        "pixel_awal": pixel_awal,
        "pixel_normal": _format_number(pixel_normal, 3),
        "image_w": image_w,
        "image_h": image_h,
        "x_center": _format_number(x_center, 2),
        "y_center": _format_number(y_center, 2),
        "box_w": _format_number(box_w, 2),
        "box_h": _format_number(box_h, 2),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "box_width_px": box_width_px,
        "box_height_px": box_height_px,
        "box_area_px": box_area_px,
        "box_area_percent": _format_number(box_area_percent, 1),
        "objectness": _format_number(objectness, 2),
        "class_probability": _format_number(class_probability, 2),
        "confidence": _format_number(confidence, 2),
        "confidence_percent": _format_number(confidence * 100, 0),
        "area_prediksi": area_prediksi,
        "area_label": area_label,
        "area_irisan": area_irisan,
        "area_gabungan": area_gabungan,
        "iou": _format_number(iou, 3),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _format_number(precision, 3),
        "recall": _format_number(recall, 3),
        "f1_score": _format_number(f1_score, 3),
        "ap_values": [_format_number(v, 2) for v in ap_values],
        "map_value": _format_number(map_value, 3),
    }
