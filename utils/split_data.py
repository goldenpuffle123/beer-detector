from pathlib import Path
import shutil
import random

def split_dataset(image_dir, label_dir, out_dir, train_ratio=0.7):
    image_path = Path(image_dir)
    label_path = Path(label_dir)
    out_path = Path(out_dir)
    
    # Get matching basenames
    image_stems = {f.stem for f in image_path.iterdir() if f.is_file()}
    label_stems = {f.stem for f in label_path.iterdir() if f.is_file()}
    matched_stems = list(image_stems & label_stems)
    
    # Split randomly
    random.shuffle(matched_stems)
    split_idx = int(len(matched_stems) * train_ratio)
    train_stems = matched_stems[:split_idx]
    val_stems = matched_stems[split_idx:]
    
    # Create output directories
    for split in ['train', 'val']:
        (out_path / "images" / split).mkdir(exist_ok=True, parents=True)
        (out_path / "labels" / split).mkdir(exist_ok=True, parents=True)
    
    # Copy files
    for stems, split in [(train_stems, 'train'), (val_stems, 'val')]:
        for stem in stems:
            img_file = next(image_path.glob(f"{stem}.*"))
            lbl_file = next(label_path.glob(f"{stem}.*"))
            shutil.copy2(img_file, out_path / "images" / split / img_file.name)
            shutil.copy2(lbl_file, out_path / "labels" / split / lbl_file.name)
    
    print(f"Split {len(train_stems)} train and {len(val_stems)} val samples.")

def validate_matching_filenames(image_dir, label_dir):
    image_stems = {f.stem for f in Path(image_dir).iterdir() if f.is_file()}
    label_stems = {f.stem for f in Path(label_dir).iterdir() if f.is_file()}
    
    only_images = image_stems - label_stems
    only_labels = label_stems - image_stems
    
    if only_images:
        print(f"Warning: These images have no matching label: {sorted(only_images)}")
    if only_labels:
        print(f"Warning: These labels have no matching image: {sorted(only_labels)}")
    if not (only_images or only_labels):
        print("All image and label filenames match.")
    
    return not (only_images or only_labels)

if __name__ == "__main__":
    split_dataset("frames", "labels", "dataset")