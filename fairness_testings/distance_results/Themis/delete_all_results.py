import os

def delete_files_in_subfolders(base_dir):
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path):
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    try:
                        os.remove(file_path)
                        print("Deleted: {}".format(file_path))
                    except Exception as e:
                        print("Failed to delete {}: {}".format(file_path, e))

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    delete_files_in_subfolders(base_dir)
