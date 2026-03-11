import os
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent             
BASE_DIR = SCRIPT_DIR.parent                             
DATA_DIR = BASE_DIR / "dataset" / "test"                    # cw1/data/test
OUTPUT_CSV = BASE_DIR / "dataset" / "test_index.csv"        # output: test_index.csv

GESTURE_CLASSES = {
    "G01_call": 0, "G02_dislike": 1, "G03_like": 2, "G04_ok": 3,
    "G05_one": 4, "G06_palm": 5, "G07_peace": 6, "G08_rock": 7,
    "G09_stop": 8, "G10_three": 9
}

def build_index():
    data_records = []

    if not DATA_DIR.exists():
        print(f"Didn't find testset folder: {DATA_DIR}")
        return

    # 遍历 test 文件夹下的 G01_call, G02_dislike ...
    for gesture_folder in DATA_DIR.iterdir():
        if not gesture_folder.is_dir():
            continue

        if gesture_folder.name not in GESTURE_CLASSES:
            continue
            
        gesture_name = gesture_folder.name
        class_label = GESTURE_CLASSES[gesture_name]
        
        # 遍历 clip folder (e.g. clip01, clip02...)
        for clip_folder in gesture_folder.iterdir():
            if not clip_folder.is_dir(): 
                continue
            clip_name = clip_folder.name
            
            rgb_dir = clip_folder / "rgb"
            annotation_dir = clip_folder / "annotation"
            depth_dir = clip_folder / "depth"  # <--- 新增: Depth 文件夹路径
            
            if not rgb_dir.exists(): 
                print(f"RGB folder not found: '{rgb_dir}'")
                continue
                
            # 遍历每一张 rgb 图片
            for rgb_file in sorted(rgb_dir.glob("*.png")):
                frame_name = rgb_file.name
                
                # 检查 Mask
                mask_file = annotation_dir / frame_name
                has_mask = mask_file.exists()
                
                # 检查 Depth
                depth_file = depth_dir / frame_name
                has_depth = depth_file.exists()
                
                record = {
                    "student": "test_subject",
                    "gesture": gesture_name,
                    "class_label": class_label,
                    "clip": clip_name,
                    "frame_name": frame_name,
                    "rgb_path": str(rgb_file.resolve()),
                    "has_mask": has_mask, 
                    "mask_path": str(mask_file.resolve()) if has_mask else None,
                    # 新增 depth 字段，与训练集的 dataset_index_split.csv 保持完全一致
                    "has_depth": has_depth,
                    "depth_path": str(depth_file.resolve()) if has_depth else None
                }
                data_records.append(record)
                    
    # 保存为 test_index.csv
    df = pd.DataFrame(data_records)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"CSV file path: {OUTPUT_CSV}")
    print(f"The total number of rgb images: {len(df)}")
    print(f"The total number of rgb images with mask annotation: {df['has_mask'].sum()}")
    print(f"The total number of rgb images with depth: {df['has_depth'].sum()}")

if __name__ == "__main__":
    build_index()