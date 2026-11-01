import os
import sys
import time
from io import BytesIO
import pandas as pd
from starlette.datastructures import UploadFile

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_APP_DIR = os.path.dirname(BACKEND_DIR)
PROJECT_ROOT = os.path.dirname(WEB_APP_DIR)
sys.path.insert(0, WEB_APP_DIR)

from backend.inference.model_loader import model_loader
from backend.api.routes import health_check, get_model_info, predict

def run_smoke_test():
    print("=" * 80)
    print("  DERMSEMBLE FASTAPI PRODUCTION BACKEND VERIFICATION & SMOKE TEST")
    print("=" * 80)

    # 1. Initialize Production Models & Manifest
    print("[1/4] Initializing Production Models & ONNX INT8 Sessions...")
    model_loader.load_all()
    print("✓ ModelLoader initialization complete.\n")

    # 2. Test Health Endpoint
    print("[2/4] Testing GET /api/v1/health ...")
    health_resp = health_check()
    print(f"  Status : {health_resp.status}")
    print(f"  Message: {health_resp.message}")
    assert health_resp.status == "ok", "Health status mismatch"
    print("✓ Health Endpoint PASSED\n")

    # 3. Test Model Info Endpoint
    print("[3/4] Testing GET /api/v1/model-info ...")
    info_resp = get_model_info()
    print(f"  Models Loaded : {info_resp.models_loaded}")
    print(f"  Versions      : {info_resp.versions}")
    print(f"  ONNX Paths    : {list(info_resp.paths.keys())}")
    assert len(info_resp.models_loaded) >= 2, "Insufficient models reported"
    print("✓ Model Info Endpoint PASSED\n")

    # 4. Test Live Diagnostic Prediction with Patient Image
    print("[4/4] Testing POST /api/v1/predict with Holdout Patient Image...")
    manifest_path = os.path.join(PROJECT_ROOT, "metadata", "splits", "test_manifest.csv")
    manifest = pd.read_csv(manifest_path)
    sample = manifest.iloc[23]  # Known Malignant Melanoma holdout case
    img_rel_path = sample["processed_path"]
    full_img_path = os.path.join(PROJECT_ROOT, img_rel_path)

    print(f"  Holdout Image ID : {sample['image_id']}")
    print(f"  Ground Truth     : {sample['unified_class']} [{sample['clinical_tier']}]")
    print(f"  Phototype        : {sample['fitzpatrick_proxy']}")

    with open(full_img_path, "rb") as f:
        raw_bytes = f.read()

    # Create Starlette UploadFile
    bio = BytesIO(raw_bytes)
    upload_file = UploadFile(filename="clinical_lesion.jpg", file=bio)

    t0 = time.time()
    pred_resp = predict(upload_file)
    total_elapsed = (time.time() - t0) * 1000

    print("-" * 80)
    print("FASTAPI PREDICTION RESPONSE VERIFICATION:")
    print(f"  • Primary Diagnosis   : {pred_resp.prediction}")
    print(f"  • Diagnostic Conf     : {pred_resp.confidence * 100:.2f}%")
    print(f"  • Inference Latency   : {pred_resp.latency_ms:.1f} ms (Pipeline roundtrip: {total_elapsed:.1f} ms)")
    print(f"  • Heatmap URL         : {pred_resp.heatmap_url}")
    print(f"  • Total Differentials : {len(pred_resp.class_probabilities)} classes evaluated")

    print("\n  Top 3 Clinical Differential:")
    for rank, p in enumerate(pred_resp.top_3_predictions, 1):
        print(f"    {rank}. {p.class_name.replace('_', ' ')}: {p.probability * 100:.2f}%")

    print("\n  Generated Clinical Triage Report:")
    for line in pred_resp.report.split("\n"):
        print(f"    | {line}")

    # Verify Heatmap File Exists on Disk
    assert pred_resp.heatmap_url is not None, "Heatmap URL is missing"
    heatmap_filename = os.path.basename(pred_resp.heatmap_url)
    heatmap_disk_path = os.path.join(WEB_APP_DIR, "backend", "static", "heatmaps", heatmap_filename)
    assert os.path.exists(heatmap_disk_path), f"Heatmap file not found at: {heatmap_disk_path}"
    file_size_kb = os.path.getsize(heatmap_disk_path) / 1024
    print(f"\n✓ Verified Grad-CAM heatmap binary exists on disk ({file_size_kb:.1f} KB)")
    print(f"  Path: {heatmap_disk_path}")

    # Clinical Correctness Check
    assert pred_resp.prediction == sample["unified_class"], "Model prediction diverged from ground truth"
    print("✓ Diagnostic Accuracy: Primary prediction accurately matched Ground Truth!")
    print("=" * 80)
    print("  ALL API ENDPOINTS & PIPELINES VERIFIED WITH 100% SUCCESS!")
    print("=" * 80)

if __name__ == "__main__":
    run_smoke_test()
