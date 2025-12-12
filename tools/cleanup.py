# write a code that removes all folders under this directory whcich are under folder names District
import os
import shutil


def cleanup_district_folders(base_dir: str) -> None:
    """
    Remove all folders under the base directory that are named 'District'.

    :param base_dir: The base directory where the cleanup should be performed.
    """
    for root, dirs, _ in os.walk(base_dir):
        for dir_name in dirs:
            if dir_name == "Zone_0":
                dir_path = os.path.join(root, dir_name)
                print(f"Removing directory: {dir_path}")
                shutil.rmtree(dir_path, ignore_errors=True)


if __name__ == "__main__":
    base_directory = os.getcwd()  # or specify your base directory
    cleanup_district_folders(base_directory)
    print("Cleanup completed.")
