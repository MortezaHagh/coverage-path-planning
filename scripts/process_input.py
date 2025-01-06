""" get data and setting from files"""
import os
import json
import copy
from typing import Optional
import yaml
import cv2 as cv
import numpy as np
from scripts.map_data import Configs


class ProcessorInputs:
    """ Read map image, data and configs """
    ns: str
    _configs: Configs
    _orig_map: Optional[np.ndarray]

    def __init__(self, map_name="Map1"):
        # init
        self.ns = "Inp"
        configs = Configs()
        self._configs = None
        self._orig_map = None

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
        self._configs = configs

        # read pgm file
        map_arr = cv.imread(pgm_file_path, cv.IMREAD_GRAYSCALE)
        self._orig_map = map_arr
        # self.visualize_map(map_arr)

    @property
    def configs(self) -> Configs:
        return copy.deepcopy(self._configs)

    @property
    def orig_map(self) -> np.ndarray:
        return copy.deepcopy(self._orig_map)

    def visualize_map(self, pgm_data) -> None:
        print(f"[{self.ns}]: visualizing")

        # check if map is loaded
        if pgm_data is None:
            print(f"[{self.ns}]: Failed to load PGM file.")
            return
        #
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
        print(f"[{self.ns}]: Image window closed.")

    def read_map_data(self, yaml_data: dict, map_data: Configs) -> None:
        """ read map yaml file and update config object """
        map_data.origin = yaml_data["origin"]
        map_data.resolution = yaml_data["resolution"]
        map_data.free_thresh = yaml_data["free_thresh"]
        map_data.occupied_thresh = yaml_data["occupied_thresh"]

    def check_directory(self, folder_path, folder_name) -> None:
        """ check if directory and map files exist """
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


if __name__ == "__main__":
    processor = ProcessorInputs("Map1")
