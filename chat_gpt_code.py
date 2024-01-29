import os
import shutil

def copy_empty_folders(src, dest):
    # Get a list of all folders in the source directory
    folders = [f for f in os.listdir(src) if os.path.isdir(os.path.join(src, f))]

    # Create empty folders in the destination directory
    for folder in folders:
        src_folder_path = os.path.join(src, folder)
        dest_folder_path = os.path.join(dest, folder)

        if not os.path.exists(dest_folder_path):
            os.makedirs(dest_folder_path)

if __name__ == "__main__":
    # Specify source and destination directories
    source_directory = "./stanford-car-dataset-by-classes-folder/car_data/car_data/train"
    destination_directory = "./predictions_dataset/car_data/test"

    # Copy empty folders from source to destination
    copy_empty_folders(source_directory, destination_directory)

    print(f"Empty folders copied from {source_directory} to {destination_directory}")
