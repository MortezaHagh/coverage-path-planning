"""Command-line entry point for the covpp package.

This module provides the `main()` function used by the console script.
It mirrors the behavior previously provided by the top-level `main.py`.
"""
import sys
import numpy as np
from covpp.gui.rectangle import Rectangle
from covpp.base.logger import MyLogger
from covpp.base.map_data import MapData
from covpp.base.configs import MapConfig, PipeLineConfig
from covpp.base.inputs_processor import InputsProcessor
from covpp.base.image_processor import MapImageProcessor
from covpp.district.district import DistrictData
from covpp.district.create_district import DistrictGen
from covpp.spa.space_analyzer import SpaceAnalyzer


def run_pipeline(map_name: str) -> None:
    """Run the complete coverage path planning pipeline.

    Args:
        map_name: Name of the map to be processed.
    """

    ns = "PipeLine"
    lg = MyLogger(ns)
    start_time = np.datetime64('now')

    #  1. process inputs: configs and map (yaml and pgm)
    inp_processor = InputsProcessor(map_name)
    orig_map = inp_processor.orig_map
    map_config: MapConfig = inp_processor.map_config

    # 2. Initial Process of the map image and create MapData object
    mp = MapImageProcessor(map_arr=orig_map, map_config=map_config)
    map_data: MapData = mp.map_data

    # 3. Select area of interest in the map and create District rectangle
    district_g = DistrictGen(map_data=map_data)
    district_rect: Rectangle = district_g.get_district_rect()

    # 4. Process Space based on map and configs (robot and workspace) - Initialize District MapData
    spa = SpaceAnalyzer(map_data=map_data, rectangle=district_rect)
    spa.create_local_map_ws_selection()

    # 5. Create DistrictData and district ZoneMapData
    district_data = DistrictData(spa)

    # 6. create zones in the district
    district_data.create_zones_hl()

    # 7. run main coverage path planning operations sequentially for all zones
    district_data.run_planning_pipeline()

    # Log execution time
    end_time = np.datetime64('now')
    lg.info(f"Total time: {end_time - start_time}")

    # Log pipeline method
    lg.info1(f"Finished, Method: {PipeLineConfig().method}")


def main():
    ns = "main"
    lg = MyLogger(ns)
    if len(sys.argv) < 2:
        lg.error("Usage: python main.py <map_name>")
        map_name = "map_fr52"  # map_fr52 Map2 Map4
    else:
        map_name = sys.argv[1]
    run_pipeline(map_name)


if __name__ == "__main__":
    main()
