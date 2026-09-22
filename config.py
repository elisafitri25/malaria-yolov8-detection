from pathlib import Path
import os








BASE_DIR = Path(__file__).resolve().parent

DATASET_ROOT = Path(
    r"C:\Users\ELISA\OneDrive\Dokumen\SKRIPSI ON THE ROAD\objek penyakit malaria\program\yolov8new2\dataset CDC Gallery"
)
INFECTED_DIR = Path(os.getenv("INFECTED_DIR", str(DATASET_ROOT / "Infected")))
AUGMENTED_DIR = Path(os.getenv("AUGMENTED_DIR", str(DATASET_ROOT / "Augmented")))
YOLO_READY_DIR = Path(os.getenv("YOLO_READY_DIR", str(DATASET_ROOT / "YOLO_Ready")))


CLASS_NAMES = ["Falciparum", "Knowlesi", "Malariae", "Ovale", "Vivax"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {idx: name for name, idx in CLASS_TO_ID.items()}

TARGET_PER_CLASS = int(os.getenv("TARGET_PER_CLASS", "1000"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))


TRAIN_RATIO = float(os.getenv("TRAIN_RATIO", "0.80"))
VAL_RATIO = float(os.getenv("VAL_RATIO", "0.10"))
TEST_RATIO = float(os.getenv("TEST_RATIO", "0.10"))


DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "yolov8s.pt")
DEFAULT_EPOCHS = int(os.getenv("DEFAULT_EPOCHS", "150"))
DEFAULT_IMGSZ = int(os.getenv("DEFAULT_IMGSZ", "832"))
DEFAULT_BATCH = int(os.getenv("DEFAULT_BATCH", "4"))
DEFAULT_DEVICE = os.getenv("DEFAULT_DEVICE", "cpu")  


LOG_DIR = BASE_DIR / "logs"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"
RUNS_DIR = BASE_DIR / "runs"

for p in [LOG_DIR, UPLOAD_DIR, RESULT_DIR, RUNS_DIR]:
    p.mkdir(parents=True, exist_ok=True)
