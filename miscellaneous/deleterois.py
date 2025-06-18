import os

def delete_roi_files(directory):
    deleted_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if "roi" in file:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    deleted_files.append(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Could not delete {file_path}: {e}")
    return deleted_files

if __name__ == "__main__":
    # Change this to your target directory
    target_directory = "DataCubes"
    delete_roi_files(target_directory)