import os
import numpy as np
from pathlib import Path
import pandas as pd

# configure dataset directory paths
SCRIPT_DIR = Path(__file__).resolve().parent                # cw1/scripts/
BASE_DIR = SCRIPT_DIR.parent                                # cw1/
DATA_DIR = BASE_DIR / "dataset" / "RGB_depth_annotations"      # Data folder: cw1/data/RGB_depth_annotations
OUTPUT_CSV = BASE_DIR / "dataset" / "dataset_index_split.csv"  # Output(.csv path)

# Map these 10 gesture to integers 0 to 9
GESTURE_CLASSES = {
    "G01_call": 0, "G02_dislike": 1, "G03_like": 2, "G04_ok": 3,
    "G05_one": 4, "G06_palm": 5, "G07_peace": 6, "G08_rock": 7,
    "G09_stop": 8, "G10_three": 9
}

def build_index():
    data_records = []

    # Dataset structure: data/RGB_Depth_annotations -> *extract layer -> ID_StudentsName -> 10 gesture -> 5 clip -> rgb images & annotations
    
    # Traverse the folders within the *extract layer
    for extracted_folder in DATA_DIR.iterdir():
        if not extracted_folder.is_dir(): 
            print(f"There is an non-folder object{extracted_folder.name}")
            continue
            
        # Traverse the ID_StudentName folder (e.g 22059968_Gerges)
        for student_folder in extracted_folder.iterdir():
            if not student_folder.is_dir(): 
                print(f"Non-folder object{student_folder.name} was found in {extracted_folder.name}")
                continue
            student_id_name = student_folder.name
            
            # Traverse the getures folder (G01 to G10)
            for gesture_folder in student_folder.iterdir():
                if not gesture_folder.is_dir():
                    print(f"Non-folder object{gesture_folder.name} was found in {student_folder.name}")
                    continue

                if gesture_folder.name not in GESTURE_CLASSES:
                    print(f"An unknown gesture category '{gesture_folder.name}' has been detected (Student: {student_id_name})")
                    continue

                gesture_name = gesture_folder.name
                class_label = GESTURE_CLASSES[gesture_name]  #
                
                # Traverse the clip folder (clip01 to clip05)
                for clip_folder in gesture_folder.iterdir():
                    if not clip_folder.is_dir(): 
                        print(f"A non-clip folder object named '{clip_folder.name}' was discovered under {gesture_name}.")
                        continue
                    clip_name = clip_folder.name
                    
                    rgb_dir = clip_folder / "rgb"
                    annotation_dir = clip_folder / "annotation"
                    depth_dir = clip_folder / "depth"
                    
                    if not rgb_dir.exists(): 
                        print(f"The RGB folder '{rgb_dir}' cannot be located. (Student: {student_id_name})")
                        continue
                        
                    # Traverse each rgb iamges (e.g. frame_xxx.png)
                    for rgb_file in sorted(rgb_dir.glob("*.png")):
                        frame_name = rgb_file.name
                        
                        mask_file = annotation_dir / frame_name
                        has_mask = mask_file.exists()
                        
                        depth_file = depth_dir / frame_name
                        has_depth = depth_file.exists()
                        
                        record = {
                            "student": student_id_name,
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
    
    unique_students = df['student'].unique()
    print(f"\n[Split Info] Detected {len(unique_students)} valid students.")
    
    # Fix the random seed to ensure the split is reproducible across runs
    np.random.seed(42)
    np.random.shuffle(unique_students)
    
    # 20 students for training, remaining 5 for validation
    train_students = unique_students[:20]
    val_students = unique_students[20:]
    
    # Add a new 'split' column to label each student as 'train' or 'val'
    df['split'] = df['student'].apply(lambda x: 'train' if x in train_students else 'val')
    
    # Save as .csv file
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"The index csv file is saved in: {OUTPUT_CSV}")
    print(f"Total images: {len(df)}")
    print(f"Train images (20 students): {len(df[df['split'] == 'train'])}")
    print(f"Val images (5 students):    {len(df[df['split'] == 'val'])}")
    print(f"Keyframes (with mask annotations): {df['has_mask'].sum()}")

 
if __name__ == "__main__":
    build_index()