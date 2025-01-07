from scripts.map_data import MapData, Configs
from scripts.inputs_processor import InputsProcessor
from scripts.image_processor import MapImageProcessor
from scripts.gui.district import District
from scripts.gui.visuals import Rectangle


def main():
    #  process inputs - configs and map_pgm
    inp_processor = InputsProcessor("Map1")
    map_arr = inp_processor.orig_map
    configs: Configs = inp_processor.configs

    # Initial Process of the map image
    mp = MapImageProcessor(map_arr, configs)
    map_data: MapData = mp.mdata

    # Create District
    district = District(map_data=map_data, configs=configs)
    district_rect: Rectangle = district.get_district_rect()
    print(district_rect.rid)

    # # Process Space
    # space = SpaceProcessor()


if __name__ == "__main__":
    main()
