import pandas as pd
import shutil
import os


# Paths
archive_dir = 'archive'
images_dir = os.path.join(archive_dir, 'images')
csv_files = ("train.csv", "val.csv", "test.csv")
data_dir = 'data'

# Create data directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)

def move_images(csv_file):
    class_labels_data = pd.read_csv(csv_file)
    files, labels = class_labels_data['filename'], class_labels_data['label']

    for file, label in zip(files, labels):
        file_name = os.path.join(images_dir, file)
        # Cretate a folder by the label name
        label_dir = os.path.join(data_dir, label)

        try:
            # Move the image to the label directory
            os.makedirs(label_dir, exist_ok=True)
            shutil.move(file_name, label_dir)
        except FileNotFoundError:
            pass

for csv_file in csv_files:
    move_images(os.path.join(archive_dir, csv_file))