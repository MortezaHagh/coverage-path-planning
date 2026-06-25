# write a code that removes all folders under this directory whcich are under folder names District
import os
import shutil


def cleanup_district_folders(base_dir: str) -> None:
    for root, dirs, files in os.walk(base_dir):
        if not "AMain" in dirs:
            continue
        for dir_name in dirs:
            if dir_name != "AMain":
                dir_path = os.path.join(root, dir_name)
                print(f"Removing directory: {dir_path}")
                shutil.rmtree(dir_path, ignore_errors=True)
        # remove all files in directory
        for file_name in os.listdir(root):
            file_path = os.path.join(root, file_name)
            if os.path.isfile(file_path):
                print(f"Removing file: {file_path}")
                os.remove(file_path)


if __name__ == "__main__":
    # base_directory = os.getcwd()  # or specify your base directory
    base_directory = "/workspaces/covpp/maps/orig_maps"
    cleanup_district_folders(base_directory)
    print("Cleanup completed.")
