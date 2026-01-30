from pathlib import Path

def rename(labels_folder: Path):
    labels = labels_folder.rglob("*.txt")
    for label in labels:
        parsed_name = str(label).split("%5C")[-1]
        label.rename(labels_folder / parsed_name)

if __name__ == "__main__":
    labels_folder = Path("labels")
    rename(labels_folder)