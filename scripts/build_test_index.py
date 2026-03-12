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

    # Traverse the folders within the test folder
    for gesture_folder in DATA_DIR.iterdir():
        if not gesture_folder.is_dir():
            continue

        if gesture_folder.name not in GESTURE_CLASSES:
            continue
            
        gesture_name = gesture_folder.name
        class_label = GESTURE_CLASSES[gesture_name]
        
        # # Traverse the clip folder (clip01 to clip05)
        for clip_folder in gesture_folder.iterdir():
            if not clip_folder.is_dir(): 
                continue
            clip_name = clip_folder.name
            
            rgb_dir = clip_folder / "rgb"
            annotation_dir = clip_folder / "annotation"
            depth_dir = clip_folder / "depth"  
            
            if not rgb_dir.exists(): 
                print(f"RGB folder not found: '{rgb_dir}'")
                continue
                
            # Traverse each rgb iamges (e.g. frame_xxx.png)
            for rgb_file in sorted(rgb_dir.glob("*.png")):
                frame_name = rgb_file.name
                
                # Check Annoations Mask is existed
                mask_file = annotation_dir / frame_name
                has_mask = mask_file.exists()
                
                # Check Depth Map is existed
                depth_file = depth_dir / frame_name
                has_depth = depth_file.exists()
                
                record = {
                    "student": "test_subject",
                    "gesture": gesture_name,
                    "class_label": class_label,
                    "clip": clip_name,
                    "frame_name": frame_name,

                    # .resolve() retrieves the image's absolute path on the computer 
                    # str() converts it into a string, which is then stored in the CSV table.
                    "rgb_path": str(rgb_file.resolve()),

                    # Record whether it has a mask (True or False)
                    #If 'has_mask' is True, store the absolute path of the mask; if False, fill in 'None'
                    "has_mask": has_mask, 
                    "mask_path": str(mask_file.resolve()) if has_mask else None,

                    "has_depth": has_depth,
                    "depth_path": str(depth_file.resolve()) if has_depth else None
                }
                data_records.append(record)
                    
    # Save as .csv file
    df = pd.DataFrame(data_records)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"CSV file path: {OUTPUT_CSV}")
    print(f"The total number of rgb images: {len(df)}")
    print(f"The total number of rgb images with mask annotation: {df['has_mask'].sum()}")
    print(f"The total number of rgb images with depth: {df['has_depth'].sum()}")

if __name__ == "__main__":
    build_index()