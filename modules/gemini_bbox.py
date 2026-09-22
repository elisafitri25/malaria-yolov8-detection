from __future__ import annotations

import json
import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

from config import CLASS_NAMES, CLASS_TO_ID
from modules.dataset_manager import IMAGE_EXTS, is_image, write_yolo_label


class BoundingBox(BaseModel):
    """Format keluaran Gemini untuk spatial understanding.

    box_2d mengikuti format dari contoh notebook Gemini:
    [y_min, x_min, y_max, x_max] dengan koordinat ternormalisasi 0..1000.
    """

    box_2d: List[int] = Field(..., min_length=4, max_length=4)
    label: str = "parasite"


@dataclass
class GeminiAnnotationSettings:
    model: str = "gemini-2.5-flash"
    limit_per_class: int = 0  # 0 berarti semua gambar
    overwrite: bool = False
    fallback_full_image: bool = False
    save_preview: bool = True
    delay_seconds: float = 0.8
    max_boxes: int = 10
    min_box_area: float = 0.0005


def _import_google_genai():
    try:
        from google import genai  # type: ignore
        from google.genai.types import GenerateContentConfig, Part, SafetySetting  # type: ignore
    except Exception as exc:  # pragma: no cover - hanya muncul di runtime user
        raise RuntimeError(
            "Library google-genai belum terpasang. Jalankan: pip install google-genai pydantic Pillow"
        ) from exc
    return genai, GenerateContentConfig, Part, SafetySetting


def _make_client():
    """Membuat client Gemini.

    Mode termudah: gunakan Gemini API key lewat GEMINI_API_KEY di file .env.
    Mode Vertex AI juga disediakan bila user memiliki Google Cloud Project.
    """
    genai, _, _, _ = _import_google_genai()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    use_vertex = os.getenv("GEMINI_USE_VERTEX", "0").strip().lower() in {"1", "true", "yes"}

    if api_key and not use_vertex:
        return genai.Client(api_key=api_key)

    project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip() or os.getenv("PROJECT_ID", "").strip()
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "global").strip() or "global"
    if use_vertex and project:
        # Beberapa versi SDK memakai vertexai=True, beberapa contoh lama memakai enterprise=True.
        try:
            return genai.Client(vertexai=True, project=project, location=location)
        except TypeError:
            return genai.Client(enterprise=True, project=project, location=location)

    raise RuntimeError(
        "GEMINI_API_KEY belum diisi di file .env. Isi GEMINI_API_KEY=... terlebih dahulu, "
        "atau gunakan mode Vertex dengan GEMINI_USE_VERTEX=1 dan GOOGLE_CLOUD_PROJECT."
    )


def _make_config(max_boxes: int):
    _, GenerateContentConfig, _, SafetySetting = _import_google_genai()
    return GenerateContentConfig(
        system_instruction=(
            "Return bounding boxes as a JSON array with labels only. Never return masks. "
            f"Limit to {max_boxes} objects. Bounding box format must be "
            "[y_min, x_min, y_max, x_max] normalized from 0 to 1000."
        ),
        temperature=0.2,
        safety_settings=[
            SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH"),
        ],
        response_mime_type="application/json",
        response_schema=list[BoundingBox],
    )


def _prompt_for_class(class_name: str) -> str:
    return f"""
You are preparing object-detection annotations for a YOLOv8 malaria microscopy dataset.
Image class folder: {class_name}.

Task:
Detect the visible Plasmodium parasite or the most relevant infected-region/object in this thin blood smear image.
Return bounding boxes around the parasite/infected region only, not around the whole image, not around the file border, and not around text/background.
Use the label exactly as "{class_name}" for every returned box.
If several parasite-like objects are visible, return up to 10 boxes.
If the object is very small, still draw a tight box around the visible parasite/infected region.
If no parasite or infected region is visible, return an empty JSON array [].
""".strip()


def _image_part(image_path: Path):
    _, _, Part, _ = _import_google_genai()
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    data = image_path.read_bytes()
    return Part.from_bytes(data=data, mime_type=mime_type)


def _parse_response(response: Any) -> List[BoundingBox]:
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        return [_coerce_box(item) for item in parsed]

    text = getattr(response, "text", "") or ""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    data = json.loads(text or "[]")
    return [_coerce_box(item) for item in data]


def _coerce_box(item: Any) -> BoundingBox:
    if isinstance(item, BoundingBox):
        return item
    if hasattr(item, "model_dump"):
        item = item.model_dump()
    elif hasattr(item, "dict"):
        item = item.dict()
    return BoundingBox(**item)


def _gemini_to_yolo(box_2d: Iterable[int], min_area: float = 0.0005) -> Optional[List[float]]:
    vals = [float(v) for v in list(box_2d)[:4]]
    if len(vals) != 4:
        return None

    y1, x1, y2, x2 = vals
    x_min, x_max = sorted([max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))])
    y_min, y_max = sorted([max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))])

    width = (x_max - x_min) / 1000.0
    height = (y_max - y_min) / 1000.0
    if width <= 0 or height <= 0 or (width * height) < min_area:
        return None

    x_center = ((x_min + x_max) / 2.0) / 1000.0
    y_center = ((y_min + y_max) / 2.0) / 1000.0
    return [x_center, y_center, width, height]


def _draw_preview(image_path: Path, boxes: List[BoundingBox], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path).convert("RGB") as im:
        width, height = im.size
        draw = ImageDraw.Draw(im)
        font = ImageFont.load_default()
        for bbox in boxes:
            y1, x1, y2, x2 = [float(v) for v in bbox.box_2d[:4]]
            x_min = int(max(0, min(1000, x1)) / 1000 * width)
            y_min = int(max(0, min(1000, y1)) / 1000 * height)
            x_max = int(max(0, min(1000, x2)) / 1000 * width)
            y_max = int(max(0, min(1000, y2)) / 1000 * height)
            draw.rectangle(((x_min, y_min), (x_max, y_max)), outline=(20, 80, 160), width=max(2, width // 250))
            draw.text((x_min + 4, max(0, y_min - 14)), bbox.label, fill=(20, 80, 160), font=font)
        im.save(out_path)


def annotate_one_image(
    image_path: Path,
    class_name: str,
    client: Any,
    config: Any,
    settings: GeminiAnnotationSettings,
) -> Dict[str, Any]:
    prompt = _prompt_for_class(class_name)
    _, _, _, _ = _import_google_genai()

    response = client.models.generate_content(
        model=settings.model,
        contents=[prompt, _image_part(image_path)],
        config=config,
    )
    gemini_boxes = _parse_response(response)

    class_id = CLASS_TO_ID[class_name]
    yolo_boxes: List[List[float]] = []
    valid_gemini_boxes: List[BoundingBox] = []
    for bbox in gemini_boxes[: settings.max_boxes]:
        yolo = _gemini_to_yolo(bbox.box_2d, min_area=settings.min_box_area)
        if yolo is None:
            continue
        yolo_boxes.append(yolo)
        valid_gemini_boxes.append(BoundingBox(box_2d=bbox.box_2d, label=class_name))

    if not yolo_boxes and settings.fallback_full_image:
        yolo_boxes = [[0.5, 0.5, 1.0, 1.0]]
        valid_gemini_boxes = [BoundingBox(box_2d=[0, 0, 1000, 1000], label=f"{class_name}_fallback_full_image")]

    if yolo_boxes:
        write_yolo_label(image_path.with_suffix(".txt"), yolo_boxes, [class_id] * len(yolo_boxes))

    return {
        "image": str(image_path),
        "label": str(image_path.with_suffix(".txt")),
        "class_name": class_name,
        "raw_boxes": [b.model_dump() for b in gemini_boxes],
        "valid_boxes": [b.model_dump() for b in valid_gemini_boxes],
        "yolo_boxes": yolo_boxes,
        "written": bool(yolo_boxes),
    }


def annotate_dataset_with_gemini(
    infected_dir: Path,
    limit_per_class: int = 5,
    overwrite: bool = False,
    fallback_full_image: bool = False,
    save_preview: bool = True,
    delay_seconds: float = 0.8,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Buat label YOLO .txt menggunakan Gemini spatial understanding.

    Fungsi ini sengaja memakai limit_per_class default kecil agar aman untuk uji coba.
    Gunakan limit_per_class=0 bila ingin memproses semua gambar.
    """
    settings = GeminiAnnotationSettings(
        model=model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        limit_per_class=int(limit_per_class),
        overwrite=overwrite,
        fallback_full_image=fallback_full_image,
        save_preview=save_preview,
        delay_seconds=float(delay_seconds),
    )
    client = _make_client()
    gen_config = _make_config(settings.max_boxes)

    review_dir = infected_dir.parent / "Gemini_BBox_Review"
    raw_json_dir = review_dir / "raw_json"
    preview_dir = review_dir / "preview"
    raw_json_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "model": settings.model,
        "infected_dir": str(infected_dir),
        "limit_per_class": settings.limit_per_class,
        "overwrite": settings.overwrite,
        "fallback_full_image": settings.fallback_full_image,
        "classes": {},
    }

    for class_name in CLASS_NAMES:
        cls_dir = infected_dir / class_name
        class_report = {"processed": 0, "written": 0, "skipped": 0, "failed": 0, "items": []}
        if not cls_dir.exists():
            class_report["status"] = "folder tidak ditemukan"
            report["classes"][class_name] = class_report
            continue

        images = [p for p in sorted(cls_dir.iterdir()) if p.is_file() and is_image(p)]
        if settings.limit_per_class > 0:
            images = images[: settings.limit_per_class]

        for image_path in images:
            label_path = image_path.with_suffix(".txt")
            if label_path.exists() and not settings.overwrite:
                class_report["skipped"] += 1
                continue
            try:
                item = annotate_one_image(image_path, class_name, client, gen_config, settings)
                class_report["processed"] += 1
                class_report["written"] += int(item["written"])
                class_report["items"].append(item)

                raw_path = raw_json_dir / class_name / f"{image_path.stem}.json"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(json.dumps(item, indent=2, ensure_ascii=False), encoding="utf-8")

                if settings.save_preview and item["valid_boxes"]:
                    preview_path = preview_dir / class_name / f"{image_path.stem}.jpg"
                    _draw_preview(image_path, [BoundingBox(**b) for b in item["valid_boxes"]], preview_path)
            except Exception as exc:
                class_report["failed"] += 1
                class_report["items"].append({"image": str(image_path), "error": str(exc)})

            if settings.delay_seconds > 0:
                time.sleep(settings.delay_seconds)

        class_report["status"] = "ok"
        report["classes"][class_name] = class_report

    report_path = review_dir / "gemini_bbox_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_path"] = str(report_path)
    report["preview_dir"] = str(preview_dir)
    return report
