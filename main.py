""" Main function to run the program """
import numpy as np
from scripts.configs import BConfigs
from scripts.map_data import MapData
from scripts.inputs_processor import InputsProcessor
from scripts.image_processor import MapImageProcessor
from scripts.gui.create_district import DistrictGen
from scripts.gui.visuals import Rectangle
from scripts.space_analyzer import SpaceAnalyzer
from scripts.district import DistrictData


def main():
    ns = "main"
    #  process inputs: configs and map
    inp_processor = InputsProcessor("Map1")
    orig_map: np.ndarray = inp_processor.orig_map
    b_configs: BConfigs = inp_processor.configs

    # Initial Process of the map image
    mp = MapImageProcessor(orig_map, b_configs)
    map_data: MapData = mp.mdata

    # Select District [rectangle] from map
    district_g = DistrictGen(map_data=map_data)
    district_rect: Rectangle = district_g.get_district_rect()

    # Process Space - Initialize District
    spa = SpaceAnalyzer(map_data=map_data, rectangle=district_rect)
    DistrictData(spa.zone_map)
    print(f"[{ns}]: zone_map.hw:", spa.zone_map.hw)


if __name__ == "__main__":
    main()
