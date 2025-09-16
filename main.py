""" Main function to run the program. """
import sys
# import cv2 as cv
import numpy as np
from scripts.base.logger import MyLogger
from scripts.base.map_data import MapData
from scripts.gui.visuals import Rectangle
from scripts.base.configs import BConfigs, DistrictConfigs
from scripts.base.inputs_processor import InputsProcessor
from scripts.base.image_processor import MapImageProcessor
# from scripts.gui.create_district import DistrictGen
from scripts.gui.create_district_simple import DistrictGen
from scripts.spa.space_analyzer import SpaceAnalyzer
from scripts.district.district import DistrictData


def run_pipeline(map_name: str) -> None:
    ns = "PipeLine"
    lg = MyLogger(ns)
    start_time = np.datetime64('now')

    #  process inputs: configs and map
    inp_processor = InputsProcessor(map_name)
    orig_map: np.ndarray = inp_processor.orig_map
    b_configs: BConfigs = inp_processor.configs
    print(" ============================================================================== ")

    # Initial Process of the map image
    mp = MapImageProcessor(orig_map, b_configs)
    map_data: MapData = mp.mdata
    print(" ============================================================================== ")

    # Select area in the map and create District rectangle
    district_g = DistrictGen(map_data=map_data)
    district_rect: Rectangle = district_g.get_district_rect()
    print(" ============================================================================== ")

    # Process Space - Initialize District Map Data
    spa = SpaceAnalyzer(map_data=map_data, rectangle=district_rect)
    spa.create_local_map_ws_selection()
    print(" ============================================================================== ")

    # Create District Data and district Zone Map Data
    district_data = DistrictData(spa)

    # create zones and map_planning for zones
    district_data.create_zones_hl()
    print(" ============================================================================== ")

    # planning sequentially for all zones
    district_data.run_coverage_planning()
    print(" ============================================================================== ")

    #
    end_time = np.datetime64('now')
    lg.info(f"Total time: {end_time - start_time}")

    #
    lg.info1(f"Method: {DistrictConfigs().method}")

    # # display map_planning
    # map_planning = district_data.dmd.map_planning
    # cv.imshow("map_planning", map_planning)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


def main():
    ns = "main"
    lg = MyLogger(ns)
    if len(sys.argv) < 2:
        lg.error("Usage: python main.py <map_name>")
        map_name = "Map2"  # map_fr52 Map2 Map4
    else:
        map_name = sys.argv[1]
    run_pipeline(map_name)


if __name__ == "__main__":
    main()
