from __future__ import annotations

import csv
import inspect
import random
import shutil
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from config import CLASS_NAMES, CLASS_TO_ID, RANDOM_SEED, TARGET_PER_CLASS
from modules.dataset_manager import (
    ImageLabelPair,
    count_dataset,
    copy_pair,
    ensure_dir,
    list_pairs,
    read_yolo_label,
    write_yolo_label,
)


def _has_param(callable_obj, name: str) -> bool:
    """Cek nama parameter agar kode tetap aman untuk Albumentations versi lama/baru."""
    try:
        return name in inspect.signature(callable_obj).parameters
    except Exception:
        return False


def _build_bbox_params(A):
    """Kompatibel lintas versi Albumentations."""
    try:
        return A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.30)
    except TypeError:
        return A.BboxParams(coord_format="yolo", label_fields=["class_labels"], min_visibility=0.30)


def _build_rotate(A):
    kwargs = {
        "limit": 30,
        "border_mode": cv2.BORDER_CONSTANT,
        "p": 0.45,
    }
    init = A.Rotate.__init__
    # Albumentations v2 memakai fill, versi lama memakai value.
    if _has_param(init, "fill"):
        kwargs["fill"] = (255, 255, 255)
    elif _has_param(init, "value"):
        kwargs["value"] = (255, 255, 255)
    return A.Rotate(**kwargs)


def _build_affine(A):
    kwargs = {
        "scale": (0.80, 1.20),
        "translate_percent": (-0.08, 0.08),
        "shear": (-8, 8),
        "rotate": (-5, 5),
        "p": 0.55,
    }
    init = A.Affine.__init__
    # Albumentations v2: border_mode + fill. Versi lama: mode + cval.
    if _has_param(init, "border_mode"):
        kwargs["border_mode"] = cv2.BORDER_CONSTANT
    elif _has_param(init, "mode"):
        kwargs["mode"] = cv2.BORDER_CONSTANT

    if _has_param(init, "fill"):
        kwargs["fill"] = (255, 255, 255)
    elif _has_param(init, "cval"):
        kwargs["cval"] = (255, 255, 255)
    return A.Affine(**kwargs)


def _build_gauss_noise(A):
    init = A.GaussNoise.__init__
    # Albumentations v2 memakai std_range dalam skala 0..1; versi lama memakai var_limit.
    if _has_param(init, "std_range"):
        kwargs = {"std_range": (0.01, 0.06), "p": 0.25}
        if _has_param(init, "mean_range"):
            kwargs["mean_range"] = (0.0, 0.0)
        return A.GaussNoise(**kwargs)
    return A.GaussNoise(var_limit=(5.0, 35.0), p=0.25)


def _build_coarse_dropout(A):
    """CoarseDropout berbeda parameter pada beberapa versi Albumentations."""
    init = A.CoarseDropout.__init__
    if _has_param(init, "num_holes_range"):
        kwargs = {
            "num_holes_range": (1, 4),
            "hole_height_range": (0.04, 0.12),
            "hole_width_range": (0.04, 0.12),
            "p": 0.30,
        }
        if _has_param(init, "fill"):
            kwargs["fill"] = 0
        return A.CoarseDropout(**kwargs)
    return A.CoarseDropout(
        max_holes=4,
        max_height=48,
        max_width=48,
        min_holes=1,
        min_height=16,
        min_width=16,
        fill_value=0,
        p=0.30,
    )


def build_transform():
    import albumentations as A

    transforms = [
        A.HorizontalFlip(p=0.50),
        A.VerticalFlip(p=0.20),
        _build_rotate(A),
        _build_affine(A),
        A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18, p=0.45),
        A.GaussianBlur(blur_limit=(3, 5), p=0.20),
        _build_gauss_noise(A),
        _build_coarse_dropout(A),
    ]
    return A.Compose(transforms, bbox_params=_build_bbox_params(A))


def _read_image_rgb(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _save_image_rgb(path: Path, image_rgb) -> None:
    ensure_dir(path.parent)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def _clear_class_dir(dst_dir: Path) -> None:
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)


def _valid_pairs_with_boxes(pairs: List[ImageLabelPair], class_id: int) -> List[ImageLabelPair]:
    """Ambil hanya gambar yang labelnya memiliki minimal satu bounding box valid."""
    valid: List[ImageLabelPair] = []
    for pair in pairs:
        bboxes, _ = read_yolo_label(pair.label_path, forced_class_id=class_id)
        if bboxes:
            valid.append(pair)
    return valid


def balance_dataset(
    infected_dir: Path,
    augmented_dir: Path,
    target_per_class: int = TARGET_PER_CLASS,
    seed: int = RANDOM_SEED,
    clean_output: bool = True,
) -> Dict[str, dict]:
    """Menyeimbangkan dataset ke target_per_class untuk setiap spesies.

    Output dibuat pada folder Augmented. Jika jumlah gambar valid lebih dari target,
    data dipilih acak. Jika kurang dari target, data asli valid disalin lalu
    diperbanyak dengan augmentasi. Bounding box YOLO ikut ditransformasikan.

    Catatan:
    - Gambar tanpa file .txt tidak dipakai.
    - Gambar dengan .txt kosong/tidak valid tidak dipakai sebagai sumber hasil akhir.
    - Nama file output mengikuti nama folder kelas, misalnya Falciparum_0001.jpg.
    """
    rng = random.Random(seed)
    transform = build_transform()
    report_rows: List[List] = []
    summary: Dict[str, dict] = {}

    augmented_dir.mkdir(parents=True, exist_ok=True)
    if clean_output:
        for cls in CLASS_NAMES:
            _clear_class_dir(augmented_dir / cls)

    for cls in CLASS_NAMES:
        src_cls_dir = infected_dir / cls
        dst_cls_dir = ensure_dir(augmented_dir / cls)
        class_id = CLASS_TO_ID[cls]

        # Hanya gambar yang punya .txt yang masuk proses.
        source_pairs = list_pairs(src_cls_dir, cls, require_label=True)
        rng.shuffle(source_pairs)

        if not source_pairs:
            summary[cls] = {
                "source": 0,
                "valid_sources": 0,
                "final": 0,
                "action": "folder kosong/tidak ditemukan atau semua gambar tanpa .txt",
                "status": "failed",
            }
            report_rows.append([cls, 0, 0, 0, "failed", "Folder kosong/tidak ditemukan atau semua gambar tanpa .txt"])
            continue

        valid_sources = _valid_pairs_with_boxes(source_pairs, class_id)
        if not valid_sources:
            summary[cls] = {
                "source": len(source_pairs),
                "valid_sources": 0,
                "final": 0,
                "action": "label kosong/tidak valid",
                "status": "failed",
                "note": "Semua file .txt kosong/tidak valid. Buka kembali di YOLOLabel dan simpan bounding box.",
            }
            report_rows.append([cls, len(source_pairs), 0, 0, "failed", "Semua file .txt kosong/tidak valid"])
            continue

        selected_originals = valid_sources[:target_per_class] if len(valid_sources) >= target_per_class else valid_sources
        final_count = 0

        # 1) Salin/undersampling data asli yang valid saja.
        for pair in selected_originals:
            final_count += 1
            dst_img = dst_cls_dir / f"{cls}_{final_count:04d}{pair.image_path.suffix.lower()}"
            copy_pair(pair, dst_img, forced_class_id=class_id)

        action = "pengurangan acak" if len(valid_sources) > target_per_class else "salin dan augmentasi"

        # 2) Augmentasi jika kurang dari target.
        attempt = 0
        max_attempts = target_per_class * 120
        while final_count < target_per_class and attempt < max_attempts:
            attempt += 1
            pair: ImageLabelPair = rng.choice(valid_sources)
            image = _read_image_rgb(pair.image_path)
            if image is None:
                continue

            bboxes, labels = read_yolo_label(pair.label_path, forced_class_id=class_id)
            if not bboxes:
                continue

            try:
                augmented = transform(image=image, bboxes=bboxes, class_labels=labels)
            except Exception:
                continue

            aug_img = augmented["image"]
            aug_bboxes = augmented.get("bboxes", [])
            aug_labels = augmented.get("class_labels", [])

            if len(aug_bboxes) == 0:
                continue

            final_count += 1
            dst_img = dst_cls_dir / f"{cls}_{final_count:04d}{pair.image_path.suffix.lower()}"
            _save_image_rgb(dst_img, aug_img)
            write_yolo_label(dst_img.with_suffix(".txt"), aug_bboxes, aug_labels)

        status = "ok" if final_count == target_per_class else "partial"
        note = "Selesai" if status == "ok" else "Belum mencapai target; cek apakah file .txt berisi bounding box valid"
        summary[cls] = {
            "source": len(source_pairs),
            "valid_sources": len(valid_sources),
            "final": final_count,
            "action": action,
            "status": status,
            "note": note,
        }
        report_rows.append([cls, len(source_pairs), len(valid_sources), final_count, status, action, note])

    report_path = augmented_dir / "_augmentation_manifest.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "source_count_with_txt", "valid_label_count", "final_count", "status", "action", "note"])
        writer.writerows(report_rows)

    summary["_report_path"] = {"path": str(report_path)}
    summary["_counts"] = count_dataset(augmented_dir)
    return summary
