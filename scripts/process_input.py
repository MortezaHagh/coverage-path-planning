""" get data and setting from files"""
import os
import sys
import json
import signal
from typing import Optional
from dataclasses import dataclass
import yaml
import cv2 as cv
import numpy as np


@dataclass
class Configs:
    def __init__(self):
        # robor configs
        self.robot_radius = 0.5
        self.safe_distance = 0.2
        # directions
        self.map_name = None
        self.map_path = None
        self.data_dir = None
        # map configs
        self.origin = None
        self.resolution = None
        self.free_thresh = None
        self.occupied_thresh = None


class ProcessorInputs:
    configs: Configs
    orig_map_arr: Optional[np.ndarray]

    def __init__(self, map_name="Map1"):
        #
        configs = Configs()
        self.configs = None
        self.orig_map_arr = None

        # define paths
        data_dir = "/home/morteza/Documents/00-CPP/00-CPP/Data"
        map_path = os.path.join(data_dir, map_name)
        yaml_file_path = os.path.join(map_path, "merge.yaml")
        json_file_path = os.path.join(map_path, "merge.json")
        pgm_file_path = os.path.join(map_path, "merge.pgm")

        # check directory - read yaml file and write json file
        self.check_directory(map_path, map_name)
        map_yaml_data = self.read_yaml(yaml_file_path)
        self.write_json(json_file_path, map_yaml_data)

        # complete configs from yaml data
        configs.map_name = map_name
        configs.map_path = map_path
        configs.data_dir = data_dir
        self.read_map_data(map_yaml_data, configs)
        self.configs = configs

        # read pgm file
        map_arr = cv.imread(pgm_file_path, cv.IMREAD_GRAYSCALE)
        self.orig_map_arr = map_arr
        # self.visualize_map(map_arr)

    def visualize_map(self, pgm_data) -> None:
        if pgm_data is None:
            print("Failed to load PGM file.")
            return
        w_name = "PGM Image"
        cv.imshow(w_name, pgm_data)
        wait_time = 1  # milliseconds
        while True:
            key = cv.waitKey(wait_time)
            # c1 = key & 0xFF == ord('q')
            c2 = cv.getWindowProperty(w_name, cv.WND_PROP_VISIBLE) < 1
            if key >= 0 or c2:  # or c1:
                break
        cv.destroyAllWindows()
        print("Image window closed.")

    def read_map_data(self, yaml_data: dict, map_data: Configs) -> None:
        map_data.origin = yaml_data["origin"]
        map_data.resolution = yaml_data["resolution"]
        map_data.free_thresh = yaml_data["free_thresh"]
        map_data.occupied_thresh = yaml_data["occupied_thresh"]

    def check_directory(self, folder_path, folder_name) -> None:
        if not os.path.isdir(folder_path):
            print(f"Directory {folder_path} does not exist")
            print(f"Map folder [{folder_name}] does not exist")
            raise FileNotFoundError(f"Directory {folder_path} does not exist")

    def read_yaml(self, file_path):
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)

    def write_json(self, file_path, data) -> None:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)


def signal_handler():  # (sig, frame)
    print('You pressed Ctrl+C!')
    cv.destroyAllWindows()
    sys.exit(0)


if __name__ == "__main__":
    processor = ProcessorInputs("Map1")
    signal.signal(signal.SIGINT, signal_handler)
