from __future__ import annotations

import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Tuple

import yaml

from config import CLASS_NAMES, CLASS_TO_ID, RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class ImageLabelPair:
    image_path: Path
    label_path: Path
    class_name: str


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_pairs(class_dir: Path, class_name: str, require_label: bool = False) -> List[ImageLabelPair]:
    """Ambil pasangan gambar dan label dari satu folder kelas.

    require_label=False:
        Semua file gambar tetap dihitung, walaupun file .txt belum ada.
        Mode ini berguna untuk fungsi hitung dataset.

    require_label=True:
        Hanya gambar yang memiliki pasangan .txt yang ikut dipakai.
        Mode ini dipakai untuk proses augmentasi agar dataset tanpa label diabaikan.
    """
    pairs: List[ImageLabelPair] = []
    if not class_dir.exists():
        return pairs

    for img in sorted(class_dir.iterdir()):
        if not img.is_file() or not is_image(img):
            continue

        txt = img.with_suffix(".txt")
        if require_label and not txt.exists():
            continue

        pairs.append(ImageLabelPair(img, txt, class_name))

    return pairs


def count_dataset(root: Path) -> Dict[str, dict]:
    result: Dict[str, dict] = {}
    for cls in CLASS_NAMES:
        cls_dir = root / cls
        pairs = list_pairs(cls_dir, cls)
        images = len(pairs)
        labels = sum(1 for p in pairs if p.label_path.exists())
        invalid = max(0, images - labels)
        result[cls] = {
            "images": images,
            "labels": labels,
            "missing_labels": invalid,
            "folder": str(cls_dir),
        }
    result["TOTAL"] = {
        "images": sum(v["images"] for k, v in result.items() if k != "TOTAL"),
        "labels": sum(v["labels"] for k, v in result.items() if k != "TOTAL"),
        "missing_labels": sum(v["missing_labels"] for k, v in result.items() if k != "TOTAL"),
        "folder": str(root),
    }
    return result


def read_yolo_label(label_path: Path, forced_class_id: int | None = None) -> Tuple[List[List[float]], List[int]]:
    """Baca label YOLO: class xc yc w h.

    forced_class_id dipakai karena data berada per folder spesies. Ini mencegah
    label lama yang semuanya class 0 ikut terbawa saat training multiclass.
    """
    bboxes: List[List[float]] = []
    class_labels: List[int] = []
    if not label_path.exists():
        return bboxes, class_labels

    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0])) if forced_class_id is None else forced_class_id
            x, y, w, h = [float(v) for v in parts[1:5]]
        except ValueError:
            continue
        # YOLO normalized harus berada pada rentang 0..1.
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            continue
        bboxes.append([x, y, w, h])
        class_labels.append(cls)
    return bboxes, class_labels


def write_yolo_label(label_path: Path, bboxes: Iterable[Iterable[float]], class_labels: Iterable[int]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for cls, box in zip(class_labels, bboxes):
        x, y, w, h = [float(v) for v in box]
        rows.append(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    label_path.write_text("".join(rows), encoding="utf-8")


def copy_pair(pair: ImageLabelPair, dst_img: Path, forced_class_id: int) -> bool:
    ensure_dir(dst_img.parent)
    ensure_dir(dst_img.with_suffix(".txt").parent)
    shutil.copy2(pair.image_path, dst_img)
    bboxes, labels = read_yolo_label(pair.label_path, forced_class_id=forced_class_id)
    write_yolo_label(dst_img.with_suffix(".txt"), bboxes, labels)
    return True


def prepare_yolo_split(augmented_dir: Path, yolo_ready_dir: Path, seed: int = RANDOM_SEED) -> Path:
    """Ubah folder per kelas menjadi struktur YOLO:

    YOLO_Ready/
      images/train, images/val, images/test
      labels/train, labels/val, labels/test
      dataset.yaml
    """
    rng = random.Random(seed)
    if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 1e-6:
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO harus = 1.0")

    split_dirs = ["images/train", "images/val", "images/test", "labels/train", "labels/val", "labels/test"]
    for sub in split_dirs:
        dst_dir = yolo_ready_dir / sub
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        ensure_dir(dst_dir)

    manifest_rows = []
    for cls in CLASS_NAMES:
        class_id = CLASS_TO_ID[cls]
        pairs = list_pairs(augmented_dir / cls, cls, require_label=True)
        pairs = [p for p in pairs if p.image_path.exists()]
        rng.shuffle(pairs)

        n = len(pairs)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        splits = {
            "train": pairs[:n_train],
            "val": pairs[n_train:n_train + n_val],
            "test": pairs[n_train + n_val:],
        }

        for split_name, split_pairs in splits.items():
            for idx, pair in enumerate(split_pairs):
                safe_name = f"{cls}_{idx:05d}_{pair.image_path.stem}{pair.image_path.suffix.lower()}"
                dst_img = yolo_ready_dir / "images" / split_name / safe_name
                dst_lbl = yolo_ready_dir / "labels" / split_name / Path(safe_name).with_suffix(".txt").name
                shutil.copy2(pair.image_path, dst_img)
                bboxes, labels = read_yolo_label(pair.label_path, forced_class_id=class_id)
                write_yolo_label(dst_lbl, bboxes, labels)
                manifest_rows.append([split_name, cls, str(dst_img), str(dst_lbl), len(bboxes)])

    dataset_yaml = yolo_ready_dir / "dataset.yaml"
    yaml_data = {
        "path": str(yolo_ready_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    dataset_yaml.write_text(yaml.safe_dump(yaml_data, sort_keys=False, allow_unicode=True), encoding="utf-8")

    manifest = yolo_ready_dir / "split_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "class", "image", "label", "bbox_count"])
        writer.writerows(manifest_rows)

    return dataset_yaml



def create_full_image_labels(root: Path, overwrite: bool = False) -> Dict[str, dict]:
    """Membuat anotasi YOLO otomatis untuk dataset berbasis folder kelas.

    Dipakai ketika dataset hanya berisi file gambar tanpa .txt. Karena sumber
    datanya sudah dipisahkan per folder kelas, setiap gambar diberi satu bbox
    penuh: class_id 0.500000 0.500000 1.000000 1.000000.

    Catatan metodologis: ini cocok jika tiap gambar merupakan crop/patch objek
    parasit atau sel yang ingin diklasifikasikan. Jika gambar adalah slide besar
    yang berisi banyak objek kecil, anotasi manual bbox tetap diperlukan.
    """
    report: Dict[str, dict] = {}
    for cls in CLASS_NAMES:
        class_id = CLASS_TO_ID[cls]
        cls_dir = root / cls
        created = 0
        skipped = 0
        images = 0
        if not cls_dir.exists():
            report[cls] = {"images": 0, "created": 0, "skipped": 0, "status": "folder tidak ditemukan"}
            continue
        for img in sorted(cls_dir.iterdir()):
            if not img.is_file() or not is_image(img):
                continue
            images += 1
            label_path = img.with_suffix(".txt")
            if label_path.exists() and not overwrite:
                skipped += 1
                continue
            write_yolo_label(label_path, [[0.5, 0.5, 1.0, 1.0]], [class_id])
            created += 1
        report[cls] = {"images": images, "created": created, "skipped": skipped, "status": "ok"}
    report["TOTAL"] = {
        "images": sum(v["images"] for k, v in report.items() if k != "TOTAL"),
        "created": sum(v["created"] for k, v in report.items() if k != "TOTAL"),
        "skipped": sum(v["skipped"] for k, v in report.items() if k != "TOTAL"),
        "status": "ok",
    }
    return report
