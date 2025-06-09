""" get data and setting from input files"""
import os
import json
import copy
import yaml
import cv2 as cv
import numpy as np
from scripts.base.configs import BConfigs


class InputsProcessor:
    """ Read input map image, data and configs """
    ns: str
    map_name: str
    paths: dict = {}
    #
    __configs: BConfigs
    __orig_map: np.ndarray

    def __init__(self, map_name="Map1"):
        # init
        self.ns = "InputsProcessor"
        self.map_name = map_name
        self.__configs = BConfigs()
        self.__orig_map = None
        self.paths["data"] = "/home/morteza/Documents/00-CPP/00-CPP/Data"

        #
        print(f" >>>>> [{self.ns}] Step: processing inputs")
        print(f"[{self.ns}]: map name: {map_name}")

        # define paths
        self.set_paths()

        # check directory
        self.check_directory()

        # read yaml file and write json file
        map_yaml_data = self.load_data(self.paths["yaml"])
        self.write_json(self.paths["json"], map_yaml_data)

        # complete configs from yaml data
        self.set_configs(map_yaml_data)

        # read pgm file
        map_arr = cv.imread(self.paths["pgm"], cv.IMREAD_GRAYSCALE)
        self.__orig_map = map_arr
        # self.visualize_map(map_arr)

    @property
    def configs(self) -> BConfigs:
        return copy.deepcopy(self.__configs)

    @property
    def orig_map(self) -> np.ndarray:
        return copy.deepcopy(self.__orig_map)

    def set_paths(self) -> None:
        map_dir = os.path.join(self.paths["data"],  self.map_name)
        self.paths["map"] = os.path.join(map_dir,  self.__configs.base_folder)
        self.paths["district"] = os.path.join(map_dir, self.__configs.district_folder)
        self.paths["yaml"] = os.path.join(self.paths["map"], "merge.yaml")
        self.paths["json"] = os.path.join(self.paths["map"], "merge.json")
        self.paths["pgm"] = os.path.join(self.paths["map"], "merge.pgm")

    def check_directory(self) -> None:
        """ check if directory and map files exist """
        if not os.path.isdir(self.paths["map"]):
            print(f"Directory {self.paths['map']} does not exist")
            raise FileNotFoundError(f"Directory {self.paths['map']} does not exist")

    def set_configs(self, yaml_data: dict) -> None:
        """ read map yaml file and update config object """
        self.__configs.map_name = self.map_name
        self.__configs.data_dir = self.paths["data"]
        self.__configs.district_dir = self.paths["district"]
        self.__configs.map_path = self.paths["map"]
        self.__configs.origin_m = yaml_data["origin"]
        self.__configs.resolution = yaml_data["resolution"]
        self.__configs.free_thresh = yaml_data["free_thresh"]
        self.__configs.occupied_thresh = yaml_data["occupied_thresh"]

    def load_data(self, file_path):
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)

    def write_json(self, file_path, data) -> None:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

    def visualize_map(self, pgm_data) -> None:
        print(f"[{self.ns}]: visualizing")

        # check if map is loaded
        if pgm_data is None:
            print(f"[{self.ns}]: Failed to load PGM file.")
            return
        #
        w_name = "PGM Image"
        cv.imshow(w_name, pgm_data)
        wait_time = 10  # milliseconds
        while True:
            key = cv.waitKey(wait_time)
            # c1 = key & 0xFF == ord('q')
            c2 = cv.getWindowProperty(w_name, cv.WND_PROP_VISIBLE) < 1
            if key >= 0 or c2:  # or c1:
                break
        cv.destroyAllWindows()


if __name__ == "__main__":
    processor = InputsProcessor("Map1")
