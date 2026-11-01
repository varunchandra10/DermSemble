import os
import shutil
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

BASE_DIR = r"c:\Users\kvcsu_ht23nk8\OneDrive\Desktop\all_Projects\BTech_Project\skin-disease\BTech_Project"
SPLITS_DIR = os.path.join(BASE_DIR, "metadata", "splits")
OUTPUT_BASE = os.path.join(BASE_DIR, "datasets", "split_data")

def copy_split(split_name, manifest_file):
    manifest_path = os.path.join(SPLITS_DIR, manifest_file)
    if not os.path.exists(manifest_path):
        print(f"[ERROR] Manifest not found: {manifest_path}")
        return 0, 0
    
    df = pd.read_csv(manifest_path)
    total_files = len(df)
    target_dir = os.path.join(OUTPUT_BASE, split_name)
    os.makedirs(target_dir, exist_ok=True)
    
    # Pre-create all class subdirectories
    classes = df['unified_class'].unique()
    for cls in classes:
        os.makedirs(os.path.join(target_dir, str(cls)), exist_ok=True)
    
    print(f"\n[INFO] Starting copy for '{split_name}' ({total_files} images across {len(classes)} classes)...")
    
    copied = 0
    missing = 0
    
    def copy_single(row):
        rel_proc_path = row['processed_path']
        src_path = os.path.join(BASE_DIR, rel_proc_path)
        cls_name = str(row['unified_class'])
        file_name = os.path.basename(src_path)
        dst_path = os.path.join(target_dir, cls_name, file_name)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            return True
        else:
            return False

    start_t = time.time()
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(copy_single, [row for _, row in df.iterrows()]))
    
    copied = sum(1 for r in results if r)
    missing = sum(1 for r in results if not r)
    elapsed = time.time() - start_t
    
    print(f"[OK] Completed '{split_name}': {copied} copied, {missing} missing in {elapsed:.1f}s")
    return copied, missing

def main():
    print("=" * 70)
    print("   PHYSICAL DATASET PARTITION GENERATOR (Train / Val / Test)")
    print("=" * 70)
    print(f"Target Output Directory: {OUTPUT_BASE}")
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    overall_start = time.time()
    
    t_copied, t_missing = copy_split("train", "train_manifest.csv")
    v_copied, v_missing = copy_split("val", "val_manifest.csv")
    test_copied, test_missing = copy_split("test", "test_manifest.csv")
    
    total_copied = t_copied + v_copied + test_copied
    total_missing = t_missing + v_missing + test_missing
    overall_elapsed = time.time() - overall_start
    
    print("\n" + "=" * 70)
    print(f"[SUMMARY] Total Copied: {total_copied} | Total Missing: {total_missing}")
    print(f"[SUMMARY] Total Time  : {overall_elapsed:.1f}s")
    print(f"[SUMMARY] Output Path : {OUTPUT_BASE}")
    print("=" * 70)

if __name__ == "__main__":
    main()

