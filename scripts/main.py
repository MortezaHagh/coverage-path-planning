from process_input import Configs, ProcessorInputs
from process_image import ImageProcessor


def main():
    processor = ProcessorInputs("Map1")
    map_arr = processor.orig_map_arr
    configs: Configs = processor.configs
    ImageProcessor(map_arr, configs)


if __name__ == "__main__":
    main()
