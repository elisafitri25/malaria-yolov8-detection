from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

from config import RUNS_DIR


def train_yolov8(
    data_yaml: Path,
    model_name: str = "yolov8s.pt",
    epochs: int = 132,
    imgsz: int = 640,
    batch: int = 4,
    device: str = "cpu",
    project_name: str = "malaria_yolov8",
) -> Dict[str, Any]:
    """Training YOLOv8 dengan Python API Ultralytics."""
    from ultralytics import YOLO

    if not Path(data_yaml).exists():
        raise FileNotFoundError("Data latih belum disiapkan. Klik tombol Siapkan Data Latih terlebih dahulu, lalu mulai pelatihan ulang.")

    model = YOLO(model_name)
    results = model.train(
        data=str(data_yaml),
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=int(batch),
        device=device,
        project=str(RUNS_DIR),
        name=project_name,
        exist_ok=True,
        patience=35,
        workers=0,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=True,
        close_mosaic=15,
        degrees=7.0,
        translate=0.10,
        scale=0.30,
        fliplr=0.5,
        flipud=0.10,
        mosaic=0.80,
        mixup=0.05,
        hsv_h=0.015,
        hsv_s=0.40,
        hsv_v=0.30,
        val=True,
        plots=True,
    )

    run_dir = Path(results.save_dir) if hasattr(results, "save_dir") else RUNS_DIR / project_name
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    return {
        "run_dir": str(run_dir),
        "best_model": str(best),
        "last_model": str(last),
        "results": str(results),
    }


def validate_model(model_path: Path, data_yaml: Path, imgsz: int = 640, device: str = "cpu") -> Dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    metrics = model.val(data=str(data_yaml), imgsz=imgsz, device=device)
    return {
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "map75": float(metrics.box.map75),
    }
