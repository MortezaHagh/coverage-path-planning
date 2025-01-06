from scripts.map_data import MapData, Configs
from scripts.process_input import ProcessorInputs
from scripts.process_image import MapImageProcessor
from scripts.gui.main_event_hl import MainEventHandler


def main():
    #  process inputs - configs and map_pgm
    processor = ProcessorInputs("Map1")
    map_arr = processor.orig_map
    configs: Configs = processor.configs

    # process map_image
    mp = MapImageProcessor(map_arr, configs)
    map_data: MapData = mp.mdata

    #
    MainEventHandler(map_data=map_data, configs=configs)


if __name__ == "__main__":
    main()
