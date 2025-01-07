from scripts.map_data import MapData, Configs
from scripts.process_input import ProcessorInputs
from scripts.process_image import MapImageProcessor
from scripts.gui.district import District
from scripts.gui.visuals import Rectangle


def main():
    #  process inputs - configs and map_pgm
    inp_processor = ProcessorInputs("Map1")
    map_arr = inp_processor.orig_map
    configs: Configs = inp_processor.configs

    # process map_image
    mp = MapImageProcessor(map_arr, configs)
    map_data: MapData = mp.mdata

    # get district
    district = District(map_data=map_data, configs=configs)
    district_rect: Rectangle = district.get_district_rect()
    print(district_rect.rid)

    #


if __name__ == "__main__":
    main()
