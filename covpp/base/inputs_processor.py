"""Input processor module for reading and validating map files and configurations."""
import os
import json
import copy
import yaml
import cv2 as cv
import numpy as np
from covpp.base.logger import MyLogger
from covpp.base.configs import MapConfig


class InputsProcessor:
    """Process input files (map.pgm and map.yaml) and create MapConfig object.

    This class handles loading map images and configuration files, validating
    directory structures, and initializing the map configuration object.

    Args:
        map_name: Name of the map to be processed (e.g., "Map1", "Map2").

    Attributes:
        map_name (str): Name of the map being processed.
        map_config (MapConfig): Map configuration object containing all map parameters.
        orig_map (np.ndarray): Original grayscale map image as numpy array.
        paths (dict): Dictionary of file and directory paths.
    """

    def __init__(self, map_name: str):
        # logging
        self.lg = MyLogger("InputsProcessor")
        self.lg.info1("Step: processing inputs")
        self.lg.info2(f"map name: {map_name}")

        # attributes
        self.map_name: str = map_name
        self.mcfg = MapConfig()
        self.__orig_map: np.ndarray = None

        # paths (directories and files)
        self.paths: dict = {}

        # create required directories
        self.set_paths()

        # check directories
        self.validate_directories()

        # read yaml file, also write json file
        map_yaml_data = self.load_yaml_file(self.paths["yaml"])

        # complete configs from yaml data
        self.update_map_configs(map_yaml_data)

        # read map image file
        self.load_eval_map_image()
        # self.visualize_map(self.__orig_map)

    @property
    def map_config(self) -> MapConfig:
        """ Return a copy of the map configuration object. """
        return copy.deepcopy(self.mcfg)

    @property
    def orig_map(self) -> np.ndarray:
        """ Return a copy of the original map image as numpy array. """
        return copy.deepcopy(self.__orig_map)

    def set_paths(self) -> None:
        """ create required directories. """
        self.paths["data"] = self.mcfg.data_dir  # base data directory [hardcoded]
        map_folder_dir = os.path.join(self.paths["data"], self.map_name)
        self.paths["map"] = os.path.join(map_folder_dir, self.mcfg.base_folder)
        self.paths["district"] = map_folder_dir
        # self.paths["district"] = os.path.join(map_folder_dir, self.mcfg.district_folder)
        self.paths["yaml"] = os.path.join(self.paths["map"], "merge.yaml")
        self.paths["json"] = os.path.join(self.paths["map"], "merge.json")
        self.paths["pgm"] = os.path.join(self.paths["map"], "merge.pgm")

    def validate_directories(self) -> None:
        """ check if directory and map files exist. """
        if not os.path.isdir(self.paths["map"]):
            self.lg.error(f"Directory {self.paths['map']} does not exist")
            raise FileNotFoundError(f"Directory {self.paths['map']} does not exist")

    def load_yaml_file(self, file_path: str) -> dict:
        """ read map yaml file and return data as dictionary. Also write json file. """
        with open(file_path, 'r', encoding='utf-8') as file:
            map_yaml_data = yaml.safe_load(file)
        self.write_json(self.paths["json"], map_yaml_data)
        return map_yaml_data

    def update_map_configs(self, yaml_data: dict) -> None:
        """ update map config object based on yaml data. """
        self.mcfg.map_name = self.map_name
        self.mcfg.district_dir = self.paths["district"]
        self.mcfg.map_dir = self.paths["map"]
        self.mcfg.origin_m = yaml_data["origin"]
        self.mcfg.resolution = yaml_data["resolution"]
        self.mcfg.free_thresh = yaml_data["free_thresh"]
        self.mcfg.occupied_thresh = yaml_data["occupied_thresh"]

    def load_eval_map_image(self):
        """ load map image from pgm file and return as numpy array. """
        map_image = cv.imread(self.paths["pgm"], cv.IMREAD_GRAYSCALE)
        if map_image is None or map_image.size == 0:
            raise FileNotFoundError("Failed to load PGM file.")
        self.__orig_map = map_image

    def write_json(self, file_path: str, data: dict) -> None:
        """ Write data to a JSON file. """
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)

    def visualize_map(self, pgm_data: np.ndarray) -> None:
        """ Visualize the loaded PGM map image using OpenCV. """
        self.lg.info("visualizing")

        # check if map is loaded
        if pgm_data is None:
            self.lg.error("Failed to load PGM file.")
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
