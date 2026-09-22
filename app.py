from __future__ import annotations

import threading
import time
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file, Response
from werkzeug.utils import secure_filename

load_dotenv()

from config import (
    BASE_DIR,
    DATASET_ROOT,
    INFECTED_DIR,
    AUGMENTED_DIR,
    YOLO_READY_DIR,
    CLASS_NAMES,
    TARGET_PER_CLASS,
    DEFAULT_MODEL,
    DEFAULT_EPOCHS,
    DEFAULT_IMGSZ,
    DEFAULT_BATCH,
    DEFAULT_DEVICE,
    UPLOAD_DIR,
    RESULT_DIR,
    RUNS_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)
from modules.dataset_manager import count_dataset, prepare_yolo_split, IMAGE_EXTS
from modules.augmentation import balance_dataset
from modules.inference import predict_image
from modules.yolo_pipeline import train_yolov8, validate_model

try:
    from modules.math_context import build_math_context
except Exception:  # halaman rumus tetap aman kalau file belum tersedia
    build_math_context = None

app = Flask(__name__)
app.secret_key = "malaria-yolov8-web-local-secret"

TASK_STATE = {
    "running": False,
    "task": "idle",
    "message": "Aplikasi siap digunakan.",
    "last_result": None,
    "error": None,
    "started_at": None,
    "finished_at": None,
}

PROPOSAL_COUNTS = {
    "Vivax": 1425,
    "Falciparum": 918,
    "Malariae": 436,
    "Ovale": 312,
    "Knowlesi": 184,
}
PROPOSAL_TOTAL = sum(PROPOSAL_COUNTS.values())
TARGET_COUNTS = {name: TARGET_PER_CLASS for name in CLASS_NAMES}
TARGET_TOTAL = sum(TARGET_COUNTS.values())
AUGMENTATION_PLAN = {}
for _name, _awal in PROPOSAL_COUNTS.items():
    _akhir = TARGET_PER_CLASS
    if _awal < _akhir:
        _aksi = f"Tambah {_akhir - _awal} citra"
        _jenis = "augmentasi"
    elif _awal > _akhir:
        _aksi = f"Pilih {_akhir} dari {_awal} citra"
        _jenis = "pengurangan acak"
    else:
        _aksi = "Sudah seimbang"
        _jenis = "tetap"
    AUGMENTATION_PLAN[_name] = {"awal": _awal, "akhir": _akhir, "aksi": _aksi, "jenis": _jenis}

SPECIES_INFO = {
    "Falciparum": {
        "slug": "falciparum",
        "title": "Plasmodium falciparum",
        "short": "Jenis malaria yang paling berbahaya dan dapat berkembang cepat bila tidak ditangani.",
        "headline": "Spesies dengan risiko klinis paling berat",
        "intro": "Plasmodium falciparum adalah salah satu penyebab malaria pada manusia. Dalam pembacaan citra, spesies ini penting dikenali karena sering dikaitkan dengan malaria berat.",
        "proposal_count": 918,
        "target_count": 1000,
        "microscopy": [
            "Pada sediaan darah tipis sering tampak bentuk cincin kecil di dalam eritrosit.",
            "Satu eritrosit dapat berisi lebih dari satu cincin parasit.",
            "Gametosit matang sering digambarkan berbentuk sabit atau pisang."
        ],
        "clinical": [
            "Dapat menyebabkan penyakit berat bila diagnosis dan terapi terlambat.",
            "Sering menjadi perhatian utama dalam sistem deteksi karena tingkat risikonya tinggi.",
            "Model membantu mengenali pola visual, tetapi keputusan medis tetap harus melalui pemeriksaan tenaga kesehatan."
        ],
        "model_note": "Jika model memberi hasil Falciparum dengan keyakinan tinggi, sistem menandai citra sebagai pola yang paling mirip dengan kelas Falciparum pada data latih.",
        "color": "#6b7fa6"
    },
    "Knowlesi": {
        "slug": "knowlesi",
        "title": "Plasmodium knowlesi",
        "short": "Malaria zoonotik yang banyak dibahas di Asia Tenggara dan dapat menyerupai spesies lain pada mikroskop.",
        "headline": "Spesies zoonotik yang perlu dibedakan dengan hati-hati",
        "intro": "Plasmodium knowlesi dikenal sebagai malaria zoonotik karena sumber alaminya berkaitan dengan primata. Pada citra mikroskopis, bentuknya dapat menyerupai spesies lain sehingga klasifikasi visual menjadi menantang.",
        "proposal_count": 184,
        "target_count": 1000,
        "microscopy": [
            "Bentuk awal dapat mirip Falciparum, sedangkan bentuk lanjut sering mirip Malariae.",
            "Karena kemiripan morfologi, label yang rapi sangat penting untuk pelatihan model.",
            "Pada dataset kecil, augmentasi membantu jumlah data tetapi tidak menggantikan variasi biologis asli."
        ],
        "clinical": [
            "Banyak dilaporkan di kawasan Asia Tenggara.",
            "Dapat berkembang cepat sehingga tetap perlu perhatian klinis.",
            "Hasil sistem sebaiknya dipakai sebagai bantuan pembacaan citra, bukan diagnosis tunggal."
        ],
        "model_note": "Jika model memilih Knowlesi, berarti pola citra paling dekat dengan contoh Knowlesi yang digunakan saat pelatihan.",
        "color": "#567f7a"
    },
    "Malariae": {
        "slug": "malariae",
        "title": "Plasmodium malariae",
        "short": "Jenis malaria yang sering memiliki parasitemia lebih rendah dan pola morfologi khas seperti bentuk pita.",
        "headline": "Spesies dengan ciri bentuk pita pada sediaan darah",
        "intro": "Plasmodium malariae merupakan salah satu spesies penyebab malaria pada manusia. Dalam citra mikroskopis, ciri bentuk tertentu dapat membantu membedakannya dari spesies lain.",
        "proposal_count": 436,
        "target_count": 1000,
        "microscopy": [
            "Bentuk trofozoit dapat tampak memanjang seperti pita melintasi eritrosit.",
            "Eritrosit biasanya tidak membesar secara mencolok.",
            "Pola bentuk roset dapat muncul pada stadium tertentu."
        ],
        "clinical": [
            "Umumnya dibahas sebagai salah satu malaria manusia dengan perjalanan yang dapat lebih lama.",
            "Jumlah parasit dalam darah dapat relatif lebih rendah dibanding Falciparum.",
            "Kualitas citra dan label tetap menentukan performa klasifikasi model."
        ],
        "model_note": "Pada hasil klasifikasi, Malariae ditampilkan bila pola visual citra paling mendekati kelas Malariae dalam data latih.",
        "color": "#7a8f58"
    },
    "Ovale": {
        "slug": "ovale",
        "title": "Plasmodium ovale",
        "short": "Jenis malaria yang dapat memiliki fase dorman di hati dan tampak pada eritrosit berbentuk oval/bergerigi.",
        "headline": "Spesies dengan ciri eritrosit oval dan potensi kekambuhan",
        "intro": "Plasmodium ovale adalah spesies malaria manusia yang dapat membentuk fase dorman di hati. Dalam pembacaan citra, bentuk eritrosit dan titik pewarnaan dapat menjadi petunjuk visual.",
        "proposal_count": 312,
        "target_count": 1000,
        "microscopy": [
            "Eritrosit yang terinfeksi dapat tampak lebih besar dan oval.",
            "Tepi eritrosit kadang terlihat tidak rata atau bergerigi.",
            "Dapat tampak titik Schüffner pada pewarnaan tertentu."
        ],
        "clinical": [
            "Dapat berkaitan dengan kekambuhan karena adanya fase dorman di hati.",
            "Sering dibahas bersama Vivax karena beberapa karakter biologisnya mirip.",
            "Klasifikasi citra membantu pemilahan awal, tetapi tetap perlu konfirmasi laboratorium."
        ],
        "model_note": "Jika sistem menampilkan Ovale, citra dianggap paling sesuai dengan pola visual kelas Ovale yang dipelajari model.",
        "color": "#9a7f60"
    },
    "Vivax": {
        "slug": "vivax",
        "title": "Plasmodium vivax",
        "short": "Spesies yang tersebar luas dan dapat kambuh karena memiliki fase dorman di hati.",
        "headline": "Spesies luas dengan kemampuan relaps",
        "intro": "Plasmodium vivax merupakan salah satu spesies malaria manusia yang penting secara global. Spesies ini dapat membentuk hipnozoit di hati sehingga infeksi dapat muncul kembali.",
        "proposal_count": 1425,
        "target_count": 1000,
        "microscopy": [
            "Eritrosit yang terinfeksi sering membesar.",
            "Dapat tampak titik Schüffner pada sediaan darah yang sesuai.",
            "Berbagai stadium parasit lebih sering terlihat pada darah perifer."
        ],
        "clinical": [
            "Sering menjadi spesies yang dominan di banyak wilayah di luar Afrika sub-Sahara.",
            "Dapat menyebabkan kekambuhan karena fase dorman di hati.",
            "Pada dataset proposal, kelas ini paling banyak sehingga perlu pengurangan acak agar seimbang."
        ],
        "model_note": "Jika model memilih Vivax, sistem membaca bahwa pola visual citra paling kuat mengarah pada contoh Vivax dalam data latih.",
        "color": "#5f78a8"
    },
}
SLUG_TO_SPECIES = {info["slug"]: name for name, info in SPECIES_INFO.items()}
SPECIES_ORDER = ["Falciparum", "Knowlesi", "Malariae", "Ovale", "Vivax"]

STATE_LOCK = threading.Lock()


def set_state(**kwargs):
    with STATE_LOCK:
        TASK_STATE.update(kwargs)


def get_state():
    with STATE_LOCK:
        return dict(TASK_STATE)


def public_dataset_status():
    counts = count_dataset(INFECTED_DIR)
    total = counts.get("TOTAL", {})
    if int(total.get("images", 0)) == 0:
        return {
            "title": "Data belum ditemukan",
            "message": "Folder Infected belum berisi data yang siap diproses.",
            "ready": False,
        }
    if int(total.get("missing_labels", 0)) > 0:
        return {
            "title": "Label belum lengkap",
            "message": "Sebagian gambar belum memiliki pasangan label.",
            "ready": False,
        }
    return {
        "title": "Data siap diproses",
        "message": "Data awal siap diproses.",
        "ready": True,
    }


def public_augmented_status():
    counts = count_dataset(AUGMENTED_DIR)
    total = counts.get("TOTAL", {})
    if int(total.get("images", 0)) == 0:
        return {
            "title": "Hasil perbanyakan belum dibuat",
            "message": "Hasil augmentasi belum tersedia.",
            "ready": False,
        }
    if int(total.get("missing_labels", 0)) > 0:
        return {
            "title": "Hasil perlu diperiksa",
            "message": "Sebagian hasil augmentasi belum memiliki label.",
            "ready": False,
        }
    return {
        "title": "Hasil perbanyakan tersedia",
        "message": "Hasil augmentasi tersedia.",
        "ready": True,
    }


def public_task_state():
    state = get_state()
    return {
        "running": state.get("running"),
        "task": state.get("task"),
        "message": state.get("message"),
        "error": state.get("error"),
        "started_at": state.get("started_at"),
        "finished_at": state.get("finished_at"),
    }


def run_background(task_name, target, *args, **kwargs):
    def _runner():
        set_state(running=True, task=task_name, message=f"{task_name} sedang berjalan...", error=None, started_at=time.strftime("%Y-%m-%d %H:%M:%S"), finished_at=None)
        try:
            result = target(*args, **kwargs)
            set_state(running=False, message=f"{task_name} selesai.", last_result=result, finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        except Exception as exc:
            set_state(running=False, error=str(exc), message=f"{task_name} gagal.", finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))

    if get_state().get("running"):
        return False
    threading.Thread(target=_runner, daemon=True).start()
    return True


@app.context_processor
def inject_globals():
    return {
        "class_names": CLASS_NAMES,
        "infected_dir": INFECTED_DIR,
        "augmented_dir": AUGMENTED_DIR,
        "yolo_ready_dir": YOLO_READY_DIR,
        "target_per_class": TARGET_PER_CLASS,
        "proposal_counts": PROPOSAL_COUNTS,
        "proposal_total": PROPOSAL_TOTAL,
        "target_counts": TARGET_COUNTS,
        "target_total": TARGET_TOTAL,
        "augmentation_plan": AUGMENTATION_PLAN,
        "species_info": SPECIES_INFO,
        "species_order": SPECIES_ORDER,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
    }


@app.route("/")
def index():
    best_model = RUNS_DIR / "malaria_yolov8" / "weights" / "best.pt"
    return render_template(
        "index.html",
        best_model=best_model,
        state=public_task_state(),
    )


@app.route("/jenis-malaria")
def species_overview():
    return render_template("species.html", species_info=SPECIES_INFO, species_order=SPECIES_ORDER)


@app.route("/jenis-malaria/<slug>")
def species_detail(slug: str):
    class_name = SLUG_TO_SPECIES.get(slug.lower())
    if not class_name:
        flash("Jenis malaria tidak ditemukan.", "warning")
        return redirect(url_for("species_overview"))
    return render_template("species_detail.html", class_name=class_name, info=SPECIES_INFO[class_name])


@app.route("/contoh-citra/<class_name>")
def sample_image(class_name: str):
    if class_name not in SPECIES_INFO:
        return Response("Kelas tidak ditemukan", status=404)
    for base in (INFECTED_DIR, AUGMENTED_DIR):
        class_dir = base / class_name
        if class_dir.exists():
            for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                files = sorted(class_dir.glob(pattern))
                if files:
                    return send_file(files[0])
    info = SPECIES_INFO[class_name]
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='900' height='560' viewBox='0 0 900 560'>
      <defs><linearGradient id='g' x1='0' x2='1'><stop stop-color='#f7fafc'/><stop offset='1' stop-color='#e9f0f7'/></linearGradient></defs>
      <rect width='900' height='560' rx='38' fill='url(#g)'/>
      <circle cx='450' cy='280' r='172' fill='none' stroke='#c9d5e3' stroke-width='3'/>
      <circle cx='390' cy='245' r='62' fill='rgba(86,127,122,.20)' stroke='{info['color']}' stroke-width='4'/>
      <circle cx='500' cy='315' r='42' fill='rgba(107,127,166,.20)' stroke='#6b7fa6' stroke-width='4'/>
      <text x='450' y='455' text-anchor='middle' font-family='Arial' font-size='34' font-weight='700' fill='#243b5a'>{info['title']}</text>
      <text x='450' y='500' text-anchor='middle' font-family='Arial' font-size='22' fill='#64748b'>Contoh citra akan muncul dari folder dataset</text>
    </svg>"""
    return Response(svg, mimetype="image/svg+xml")


@app.route("/dataset")
def dataset():
    return render_template(
        "dataset.html",
        data_status=public_dataset_status(),
        augmented_status=public_augmented_status(),
        state=public_task_state(),
    )


@app.post("/run-augmentation")
def run_augmentation_route():
    target_count = int(request.form.get("target", TARGET_PER_CLASS))
    clean_output = request.form.get("clean_output", "on") == "on"
    ok = run_background(
        "Penyeimbangan dan perbanyakan data",
        balance_dataset,
        INFECTED_DIR,
        AUGMENTED_DIR,
        target_count,
        42,
        clean_output,
    )
    flash("Proses perbanyakan data dimulai. Tunggu sampai statusnya selesai." if ok else "Masih ada proses lain yang berjalan.", "info")
    return redirect(url_for("dataset"))


@app.post("/prepare-yolo")
def prepare_yolo_route():
    ok = run_background("Penyiapan data YOLO", prepare_yolo_split, AUGMENTED_DIR, YOLO_READY_DIR, 42)
    flash("Data sedang disiapkan agar bisa dibaca oleh YOLOv8." if ok else "Masih ada proses lain yang berjalan.", "info")
    return redirect(url_for("dataset"))


@app.route("/train", methods=["GET", "POST"])
def train():
    flash(
        "Menu latih model tidak ditampilkan di website. Jalankan pelatihan ulang melalui terminal VS Code memakai tools/train_from_terminal.py.",
        "info",
    )
    return redirect(url_for("index"))


@app.post("/validate")
def validate_route():
    model_path = Path(request.form.get("model_path", RUNS_DIR / "malaria_yolov8" / "weights" / "best.pt"))
    data_yaml = YOLO_READY_DIR / "dataset.yaml"
    device = request.form.get("device", DEFAULT_DEVICE)
    imgsz = int(request.form.get("imgsz", DEFAULT_IMGSZ))
    try:
        metrics = validate_model(model_path, data_yaml, imgsz=imgsz, device=device)
        flash(f"Validasi selesai: mAP50-95={metrics['map50_95']:.4f}, mAP50={metrics['map50']:.4f}", "success")
    except Exception as exc:
        flash(f"Validasi gagal: {exc}", "danger")
    return redirect(url_for("dataset"))


def _clamp_float(value, default: float, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def _clamp_int(value, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def resolve_model_path(raw_model_path: str | None = None) -> Path:
    candidates = []
    if raw_model_path and str(raw_model_path).strip():
        candidates.append(Path(str(raw_model_path).strip()))

    candidates.extend([
        RUNS_DIR / "malaria_yolov8" / "weights" / "best.pt",
        BASE_DIR / "best.pt",
        BASE_DIR / "last.pt",
    ])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0] if candidates else RUNS_DIR / "malaria_yolov8" / "weights" / "best.pt"


def _display_path(path: Path) -> str:
    """Tampilkan path secara aman untuk UI tanpa membuka nama folder lokal pengguna."""
    path = Path(path)
    for base in (BASE_DIR, DATASET_ROOT):
        try:
            rel = path.relative_to(base)
            return str(rel).replace("/", "\\")
        except ValueError:
            continue
    return path.name


def _file_info(path: Path) -> dict:
    path = Path(path)
    exists = path.exists()
    size_mb = round(path.stat().st_size / (1024 * 1024), 2) if exists and path.is_file() else 0
    return {
        "path": str(path),
        "display_path": _display_path(path),
        "exists": exists,
        "size_mb": size_mb,
        "name": path.name,
    }


@app.route("/dashboard")
def dashboard():
    model_path = resolve_model_path(None)
    dataset_yaml = YOLO_READY_DIR / "dataset.yaml"
    return render_template(
        "dashboard.html",
        state=public_task_state(),
        data_status=public_dataset_status(),
        augmented_status=public_augmented_status(),
        model_info=_file_info(model_path),
        yaml_info=_file_info(dataset_yaml),
        best_model=model_path,
        proposal_total=PROPOSAL_TOTAL,
        target_total=TARGET_TOTAL,
    )


def _allowed_image_filename(filename: str | None) -> bool:
    if not filename:
        return False
    return Path(filename).suffix.lower() in IMAGE_EXTS


def _reject_image_message() -> str:
    return "File ditolak. Unggah hanya gambar dengan format JPG, JPEG, PNG, BMP, TIFF, atau WEBP."


def _verify_saved_image(image_path: Path) -> None:
    try:
        with Image.open(image_path) as img:
            img.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(_reject_image_message()) from exc


def _write_prediction_report(report_path: Path, original_filename: str, result: dict) -> None:
    lines = []
    lines.append("LAPORAN RINGKAS HASIL KLASIFIKASI MALARIA YOLOv8")
    lines.append("=" * 58)
    lines.append(f"Waktu proses      : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"File input        : {original_filename}")
    lines.append(f"Model             : {result.get('model_path', '-')}")
    lines.append(f"Ukuran gambar     : {result.get('image_width', '-')} x {result.get('image_height', '-')}")
    lines.append(f"Confidence utama  : {round(float(result.get('requested_conf', 0)) * 100, 2)}%")
    lines.append(f"Confidence dipakai : {round(float(result.get('used_conf', 0)) * 100, 2)}%")
    lines.append(f"Kelas utama       : {result.get('classification_label', '-')}")
    lines.append(f"Keyakinan         : {result.get('classification_confidence', 0)}%")
    lines.append(f"Jumlah deteksi    : {result.get('count', 0)}")
    lines.append("")
    lines.append("Catatan:")
    lines.append(result.get("classification_message", "-"))
    lines.append("")
    lines.append("Detail deteksi:")
    detections = result.get("detections") or []
    if not detections:
        lines.append("- Tidak ada area yang terdeteksi.")
    else:
        for idx, item in enumerate(detections, start=1):
            lines.append(
                f"{idx}. {item.get('class_name', '-')} | "
                f"confidence {round(float(item.get('confidence', 0)) * 100, 2)}% | "
                f"area {item.get('area_pct', 0)}% | "
                f"bbox ({item.get('x1')}, {item.get('y1')}, {item.get('x2')}, {item.get('y2')})"
            )
    lines.append("")
    lines.append("Hasil ini bersifat bantuan analisis citra, bukan pengganti diagnosis dokter atau pemeriksaan laboratorium resmi.")
    report_path.write_text("\n".join(lines), encoding="utf-8")


@app.get("/download-result/<path:filename>")
def download_result(filename: str):
    safe_filename = secure_filename(Path(filename).name)
    file_path = RESULT_DIR / safe_filename
    if not safe_filename or not file_path.exists():
        flash("File hasil klasifikasi tidak ditemukan.", "warning")
        return redirect(url_for("predict"))
    return send_file(file_path, as_attachment=True, download_name=safe_filename)


@app.get("/download-hasil/<kind>")
def download_hasil(kind: str):
    run_dir = RUNS_DIR / "malaria_yolov8"
    allowed = {
        "best-model": run_dir / "weights" / "best.pt",
        "last-model": run_dir / "weights" / "last.pt",
        "hasil-training": run_dir / "results.csv",
        "split-manifest": YOLO_READY_DIR / "split_manifest.csv",
        "dataset-yaml": YOLO_READY_DIR / "dataset.yaml",
    }
    file_path = allowed.get(kind)
    if file_path is None or not file_path.exists():
        flash("File yang ingin diunduh belum tersedia. Jalankan proses penyusunan data atau pelatihan melalui terminal terlebih dahulu.", "warning")
        return redirect(url_for("dataset"))
    return send_file(file_path, as_attachment=True, download_name=file_path.name)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    default_model_path = resolve_model_path(None)

    if request.method == "POST":
        file = request.files.get("image")
        model_path = resolve_model_path(request.form.get("model_path"))
        conf = _clamp_float(request.form.get("conf", "0.25"), 0.25, 0.01, 1.00)
        iou = _clamp_float(request.form.get("iou", "0.45"), 0.45, 0.01, 1.00)
        imgsz = _clamp_int(request.form.get("imgsz", DEFAULT_IMGSZ), DEFAULT_IMGSZ, 128, 2048)

        if not file or file.filename == "":
            flash("Pilih gambar terlebih dahulu.", "warning")
            return redirect(url_for("predict"))

        if not _allowed_image_filename(file.filename):
            flash(_reject_image_message(), "danger")
            return redirect(url_for("predict"))

        original_filename = file.filename
        filename = secure_filename(file.filename)
        if not filename:
            filename = f"upload_{int(time.time())}.jpg"

        safe_name = f"{int(time.time())}_{filename}"
        upload_path = UPLOAD_DIR / safe_name
        file.save(upload_path)

        try:
            _verify_saved_image(upload_path)
        except ValueError as exc:
            upload_path.unlink(missing_ok=True)
            flash(str(exc), "danger")
            return redirect(url_for("predict"))

        output_path = RESULT_DIR / f"pred_{safe_name}"

        try:
            result = predict_image(
                model_path=model_path,
                image_path=upload_path,
                output_path=output_path,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                fallback_conf=0.01,
            )

            if result.get("detections"):
                top = max(result["detections"], key=lambda d: d.get("confidence", 0))
                result["top_detection"] = top
                result["classification_label"] = top.get("class_name", "Tidak diketahui")
                result["classification_confidence"] = round(float(top.get("confidence", 0)) * 100, 2)

                if result.get("used_fallback"):
                    result["classification_message"] = (
                        "Model belum menemukan objek pada batas confidence utama. "
                        "Sistem lalu mengecek ulang dengan batas sangat rendah dan menemukan kandidat lemah. "
                        "Gunakan hasil ini sebagai indikasi awal, bukan keputusan final."
                    )
                else:
                    result["classification_message"] = (
                        "Citra paling kuat mengarah pada kelas ini berdasarkan area objek yang berhasil dideteksi model."
                    )
            else:
                result["top_detection"] = None
                result["classification_label"] = "Tidak terdeteksi"
                result["classification_confidence"] = 0
                result["classification_message"] = (
                    "Model belum menemukan objek, bahkan setelah dicek ulang dengan confidence rendah. "
                    "Pastikan gambar berupa citra mikroskopis yang jelas, model sudah dilatih dengan benar, "
                    "dan label bounding box data latih sudah tepat."
                )

            result["input_url"] = url_for("static", filename=f"uploads/{safe_name}")
            result["output_url"] = url_for("static", filename=f"results/{output_path.name}")
            result["download_url"] = url_for("download_result", filename=output_path.name)
            result["model_path"] = _display_path(model_path)
            result["requested_conf"] = conf
            result["iou"] = iou
            result["imgsz"] = imgsz

            report_path = RESULT_DIR / f"laporan_{Path(safe_name).stem}.txt"
            _write_prediction_report(report_path, original_filename, result)
            result["report_download_url"] = url_for("download_result", filename=report_path.name)

            flash("Pengujian gambar selesai.", "success")
        except Exception as exc:
            message = str(exc)
            if "OpenCV" in message or "cv2" in message or "Gambar gagal dibaca" in message:
                flash(_reject_image_message(), "danger")
            else:
                flash(f"Prediksi gagal: {exc}", "danger")

    return render_template("predict.html", result=result, defaults={
        "best_model": default_model_path,
        "best_model_display": _display_path(default_model_path),
        "imgsz": DEFAULT_IMGSZ,
        "conf": 0.25,
        "iou": 0.45,
    })


@app.route("/math")
def math_page():
    context = build_math_context() if build_math_context else {}
    return render_template("math.html", **context)


@app.route("/blackbox-testing")
def blackbox_testing():
    return render_template("blackbox.html")


@app.get("/status")
def status():
    return jsonify(public_task_state())


@app.get("/api/status-ringkas")
def api_status_ringkas():
    return jsonify({
        "data": public_dataset_status(),
        "hasil_perbanyakan": public_augmented_status(),
        "proses": public_task_state(),
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
