"""
Train YOLOv8 Cricket Bat Detector
===================================
Trains a specialized YOLOv8 model to detect cricket bats.

USAGE:
    python train_yolov8_bat_detector.py

DATASET STRUCTURE (Roboflow YOLOv8 export — use as-is):
    datasets/cricket_bat/
    ├── data.yaml
    ├── train/
    │   ├── images/   (*.jpg / *.png)
    │   └── labels/   (*.txt  YOLO format: class cx cy w h)
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/         (optional, not used during training)
        ├── images/
        └── labels/

HOW TO SET UP:
    1. Download from Roboflow in "YOLOv8" format
    2. Rename/move the extracted folder to:  datasets/cricket_bat/
    3. Run this script

OUTPUT:
    features/SHOT_CLASSIFICATION_SYSTEM/trained_models/yolov8_bat_detector/best_model.pt
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import yaml

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  — only edit these lines before running
# ─────────────────────────────────────────────────────────────────────────────

# Path to your dataset folder (the one you downloaded from Roboflow)
# Put your dataset at:  datasets/cricket_bat/
THIS_FILE    = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent.parent.parent   # shot-classification/

DATASET_ROOT = PROJECT_ROOT / "datasets" / "cricket_bat"
DATASET_YAML = DATASET_ROOT / "data.yaml"

BASE_MODEL   = "yolov8n.pt"  # yolov8n=fastest  yolov8s=balanced  yolov8m=most accurate
EPOCHS       = 100
IMG_SIZE     = 640           # matches your Roboflow export (640x640)
BATCH_SIZE   = 4            # reduce to 8 if you get CUDA out-of-memory
DEVICE       = None           # "0" for GPU,  "cpu" for CPU
PROJECT_NAME = "cricket_bat_detection"
RUN_NAME     = "bat_detector_v1"

# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("features/SHOT_CLASSIFICATION_SYSTEM/trained_models/yolov8_bat_detector")


def verify_dataset() -> bool:
    """
    Verify the Roboflow-exported dataset is correctly placed and structured.
    Supports both Roboflow layout (train/valid/test at root) and
    the old expected layout (images/train  images/val).
    """
    print("🔍 Verifying dataset...")

    if not DATASET_YAML.exists():
        print(f"❌ data.yaml not found at: {DATASET_YAML}")
        print()
        print("   Fix: put your Roboflow dataset folder at  datasets/cricket_bat/")
        print("   The folder should contain data.yaml, train/, valid/, test/")
        return False

    # Read and display data.yaml
    with open(DATASET_YAML) as f:
        data = yaml.safe_load(f)

    nc    = data.get("nc",    0)
    names = data.get("names", [])
    print(f"   nc:    {nc}")
    print(f"   names: {names}")

    if nc < 1:
        print("❌ data.yaml has nc=0. Something is wrong with your dataset.")
        return False

    # Check split folders — Roboflow uses train/valid, not images/train
    roboflow_layout = (DATASET_ROOT / "train" / "images").exists()
    classic_layout  = (DATASET_ROOT / "images" / "train").exists()

    if roboflow_layout:
        print("   ✅ Roboflow layout detected  (train/images  valid/images)")
        for split in ("train", "valid"):
            img_dir   = DATASET_ROOT / split / "images"
            label_dir = DATASET_ROOT / split / "labels"
            imgs   = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
            labels = list(label_dir.glob("*.txt"))
            print(f"   {split}: {len(imgs)} images, {len(labels)} labels")
    elif classic_layout:
        print("   ✅ Classic layout detected  (images/train  images/val)")
    else:
        print(f"❌ Could not find train/images or images/train inside {DATASET_ROOT}")
        print("   Make sure you extracted the Roboflow zip into datasets/cricket_bat/")
        return False

    # Fix data.yaml paths to be absolute so YOLO can find them from any cwd
    _fix_data_yaml_paths(data, roboflow_layout)

    print("✅ Dataset looks good!\n")
    return True


def _fix_data_yaml_paths(data: dict, roboflow_layout: bool):
    """
    Roboflow data.yaml sometimes uses relative paths like '../train/images'
    which break when running from a different working directory.
    We rewrite them to absolute paths to be safe.
    """
    abs_root = DATASET_ROOT.resolve()

    if roboflow_layout:
        data["path"]  = str(abs_root)
        data["train"] = "train/images"
        data["val"]   = "valid/images"   # Roboflow uses 'valid', not 'val'
        if (abs_root / "test" / "images").exists():
            data["test"] = "test/images"
    else:
        data["path"]  = str(abs_root)
        data["train"] = "images/train"
        data["val"]   = "images/val"

    # Write back
    with open(DATASET_YAML, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"   📝 data.yaml paths updated to absolute paths")


def train_bat_detector():
    """Train the cricket bat detection model."""
    print("=" * 60)
    print("  🏏 YOLOv8 Cricket Bat Detector — Training")
    print("=" * 60)

    if not verify_dataset():
        return None

    print(f"📦 Loading base model: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)

    print(f"\n🚀 Starting training...")
    print(f"   Dataset:    {DATASET_YAML}")
    print(f"   Epochs:     {EPOCHS}")
    print(f"   Image size: {IMG_SIZE}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Device:     {DEVICE}")
    print()

    results = model.train(
        data=str(DATASET_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device="cpu",
        project=PROJECT_NAME,
        name=RUN_NAME,
        resume=False,              # resume if interrupted
        save=True,
        save_period=1,            # save checkpoint every epoch
        val=True,
        plots=True,
        verbose=True,

        # ── Augmentation ─────────────────────────────────────
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=20.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,

        # ── Training settings ────────────────────────────────
        patience=30,

        # ── Optimizer ────────────────────────────────────────
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
    )

    # ── Copy best.pt to our trained_models folder ─────────────────────────────
    best_pt = Path(PROJECT_NAME) / RUN_NAME / "weights" / "best.pt"

    if best_pt.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        dest = OUTPUT_DIR / "best_model.pt"
        shutil.copy2(best_pt, dest)
        print(f"\n✅ Best model saved → {dest}")

        # 🧹 Remove training checkpoints to save space
        weights_dir = Path(PROJECT_NAME) / RUN_NAME / "weights"
        for file in weights_dir.glob("*.pt"):
            if file.name != "best.pt":
                file.unlink()

        print("🧹 Temporary checkpoints deleted.")

        # Save a quick training summary
        metrics = results.results_dict if hasattr(results, "results_dict") else {}
        summary_path = OUTPUT_DIR / "training_summary.txt"
        with open(summary_path, "w") as f:
            f.write("Cricket Bat Detector — Training Summary\n")
            f.write("=" * 40 + "\n")
            f.write(f"Base model:  {BASE_MODEL}\n")
            f.write(f"Epochs:      {EPOCHS}\n")
            f.write(f"Image size:  {IMG_SIZE}\n")
            f.write(f"Batch size:  {BATCH_SIZE}\n\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")
        print(f"📄 Training summary → {summary_path}")

        return str(dest)
    else:
        print(f"⚠️  best.pt not found at expected path: {best_pt}")
        return None


def validate_trained_model(model_path: str):
    """Run validation on the saved best model and print metrics."""
    print(f"\n🔍 Validating: {model_path}")
    model = YOLO(model_path)
    metrics = model.val(
        data=str(DATASET_YAML),
        imgsz=IMG_SIZE,
        device=DEVICE,
        split="val",        # uses valid/ split
    )
    print(f"\n📊 Validation Results:")
    print(f"   mAP50:     {metrics.box.map50:.4f}   ← aim for > 0.70")
    print(f"   mAP50-95:  {metrics.box.map:.4f}")
    print(f"   Precision: {metrics.box.mp:.4f}")
    print(f"   Recall:    {metrics.box.mr:.4f}")

    if metrics.box.map50 < 0.60:
        print("\n⚠️  mAP50 < 0.60 — model may underperform.")
        print("   Tips to improve:")
        print("   • Increase EPOCHS to 150")
        print("   • Use a larger base model: yolov8s.pt or yolov8m.pt")
        print("   • Check label quality by visually inspecting some images")
    elif metrics.box.map50 >= 0.80:
        print("\n🎉 Excellent! mAP50 >= 0.80 — Tier 1 detection should work well.")


if __name__ == "__main__":
    model_path = train_bat_detector()
    if model_path:
        validate_trained_model(model_path)
        print("\n✅ All done!")
        print(f"   Model ready at: {model_path}")
        print()
        print("   Next steps:")
        print("   1. Check mAP50 above — aim for > 0.70")
        print("   2. Replace ball_detector.py with the new 4-tier version")
        print("   3. Tier 1 will now use: custom ball model + custom bat model")