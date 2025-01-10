from scripts.configs import BConfigs
from scripts.map_data import MapData
from scripts.inputs_processor import InputsProcessor
from scripts.image_processor import MapImageProcessor
from scripts.gui.district import District
from scripts.gui.visuals import Rectangle
from scripts.space_analyzer import SpaceAnalyzer


def main():
    #  process inputs - configs and map_pgm
    inp_processor = InputsProcessor("Map1")
    map_arr = inp_processor.orig_map
    bconfigs: BConfigs = inp_processor.configs

    # Initial Process of the map image
    mp = MapImageProcessor(map_arr, bconfigs)
    map_data: MapData = mp.mdata

    # Create District
    district = District(map_data=map_data)
    district_rect: Rectangle = district.get_district_rect()

    # Process Space
    SpaceAnalyzer(map_data=map_data, bconfigs=bconfigs, district_rect=district_rect)


if __name__ == "__main__":
    main()
