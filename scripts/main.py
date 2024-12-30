from process_input import Configs, ProcessorInputs
from process_image import MapImageProcessor


def main():
    #  process inputs - configs and map_pgm
    processor = ProcessorInputs("Map1")
    map_arr = processor.orig_map_arr
    configs: Configs = processor.configs

    # process map_image
    MapImageProcessor(map_arr, configs)


if __name__ == "__main__":
    main()
