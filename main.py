""" Main function to run the program """
import sys
import numpy as np
from scripts.base.configs import BConfigs
from scripts.base.map_data import MapData
from scripts.gui.visuals import Rectangle
from scripts.base.inputs_processor import InputsProcessor
from scripts.base.image_processor import MapImageProcessor
from scripts.gui.create_district import DistrictGen
from scripts.spa.space_analyzer import SpaceAnalyzer
from scripts.district.district import DistrictData
from scripts.district.coverage_path_planning import CPPSimple


def main():
    ns = "main"
    if len(sys.argv) < 2:
        print(f"[{ns}] Usage: python main.py <map_name>")
        map_name = "Map2"
    else:
        map_name = sys.argv[1]

    #  process inputs: configs and map
    inp_processor = InputsProcessor(map_name)
    orig_map: np.ndarray = inp_processor.orig_map
    b_configs: BConfigs = inp_processor.configs

    # Initial Process of the map image
    mp = MapImageProcessor(orig_map, b_configs)
    map_data: MapData = mp.mdata

    # Select District [rectangle] from map
    district_g = DistrictGen(map_data=map_data)
    district_rect: Rectangle = district_g.get_district_rect()

    # Process Space - Initialize District Map Data
    spa = SpaceAnalyzer(map_data=map_data, rectangle=district_rect)
    spa.create_local_map_ws_selection()

    # Create District Data and Zone Map Data
    district_data = DistrictData(spa)
    # district_data.create_zones()

    # create zones high level
    district_data.create_zones_hl()

    # coverage path planning
    CPPSimple(district_data)

    # add os path to map_hl
    district_data.add_os_path_map_hl()

    #
    print(f"[{ns}]: resolution:", spa.orig_map_data.up_resolution)


if __name__ == "__main__":
    main()
