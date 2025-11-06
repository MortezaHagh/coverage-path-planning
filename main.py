
""" Main module to run the coverage path planning pipeline. """
import sys
# import cv2 as cv
import numpy as np
from scripts.gui.rectangle import Rectangle
from scripts.base.logger import MyLogger
from scripts.base.map_data import MapData
from scripts.base.configs import MapConfig, PipeLineConfig
from scripts.base.inputs_processor import InputsProcessor
from scripts.base.image_processor import MapImageProcessor
from scripts.district.district import DistrictData
from scripts.district.create_district_simple import DistrictGen
# from scripts.district.create_district import DistrictGen
from scripts.spa.space_analyzer import SpaceAnalyzer


def run_pipeline(map_name: str) -> None:
    """Run the complete coverage path planning pipeline.

    Args:
        map_name: Name of the map to be processed.

    Pipeline Steps:
        1. Process inputs (configs and map image)
        2. Process map image and create MapData
        3. Create district rectangle
        4. Analyze workspace and initialize district MapData
        5. Create DistrictData and zone MapData
        6. Create zones
        7. Run coverage planning for all zones
    """

    ns = "PipeLine"
    lg = MyLogger(ns)
    start_time = np.datetime64('now')

    #  1. process inputs: configs and map (yaml and pgm)
    inp_processor = InputsProcessor(map_name)
    orig_map: np.ndarray = inp_processor.orig_map
    map_config: MapConfig = inp_processor.map_config
    print(" ============================================================================== ")

    # 2. Initial Process of the map image and create MapData object
    mp = MapImageProcessor(map_arr=orig_map, map_config=map_config)
    map_data: MapData = mp.map_data
    print(" ============================================================================== ")

    # 3. Select area of interest in the map and create District rectangle
    district_g = DistrictGen(map_data=map_data)
    district_rect: Rectangle = district_g.get_district_rect()
    print(" ============================================================================== ")

    # 4. Process Space based on map and configs (robot and workspace) - Initialize District MapData
    spa = SpaceAnalyzer(map_data=map_data, rectangle=district_rect)
    spa.create_local_map_ws_selection()
    print(" ============================================================================== ")

    # 5. Create DistrictData and district ZoneMapData
    district_data = DistrictData(spa)

    # 6. create zones in the district
    district_data.create_zones_hl()
    print(" ============================================================================== ")

    # 7. run main coverage path planning operations sequentially for all zones
    district_data.run_coverage_planning()
    print(" ============================================================================== ")

    # Log execution time
    end_time = np.datetime64('now')
    lg.info(f"Total time: {end_time - start_time}")

    # Log pipeline method
    lg.info1(f"Method: {PipeLineConfig().method}")

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
        map_name = "Map4"  # map_fr52 Map2 Map4
    else:
        map_name = sys.argv[1]
    run_pipeline(map_name)


if __name__ == "__main__":
    main()
