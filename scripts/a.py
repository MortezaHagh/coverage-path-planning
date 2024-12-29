""" get data and setting from files"""
import os
import json
import yaml
import cv2 as cv


class MapData:
    def __init__(self):
        self.name = None
        self.path = None
        self.data_dir = None
        #
        self.origin = None
        self.resolution = None
        self.free_thresh = None
        self.occupied_thresh = None
        #
        self.pgm = []


class ProcessorInputs:
    def __init__(self, map_name):
        # define paths
        data_dir = "/home/morteza/Documents/00-CPP/00-CPP/Data"
        map_path = os.path.join(data_dir, map_name)
        yaml_file_path = os.path.join(map_path, "merge.yaml")
        json_file_path = os.path.join(map_path, "merge.json")

        # check directory - read yaml file and write json file
        self.check_directory(map_path, map_name)
        map_yaml_data = self.read_yaml(yaml_file_path)
        self.write_json(json_file_path, map_yaml_data)

        # create map data object
        map_data = MapData()
        map_data.name = map_name
        map_data.path = map_path
        map_data.data_dir = data_dir
        self.read_map_data(map_yaml_data, map_data)

    def read_map_data(self, yaml_data: dict, map_data: MapData) -> None:
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


if __name__ == "__main__":
    processor = ProcessorInputs("Map1")
