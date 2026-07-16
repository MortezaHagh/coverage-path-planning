""" Run all coverage path planning methods on all maps. """
from covpp.base.configs import PipeLineConfig


map_names = ["MapT" + str(i) for i in range(1, 12)]
map_names += ["map_u1", "map_u3", "map_u4"]
# map_names = ["Map_fr52", "Map4"]
# map_names = ["MapT6"]

if __name__ == "__main__":

    # seg_methods = ["simple", "skeleton", "morphological", "distance_tf"]
    seg_methods = ["morphological"]
    # covpp_methods = ["boust", "contour", "grid_energy_min"]
    covpp_methods = ["contour"]
    # rects = [False, True]
    rects = [False]

    # covpp_method = ["boust"]

    dcfg = PipeLineConfig()
    dcfg.set_print(False)
    name_suffix = ""  # e1 e2

    # run for all maps
    for map_name in map_names:

        # # set start cell idx based on map
        # if map_name == "map_fr52":
        #     S_CELL_IDX = 3
        # else:
        #     S_CELL_IDX = 0

        # run for all segmentation methods
        for segm_method in seg_methods:

            # run for all coverage path planning methods
            for cpp_method in covpp_methods:

                # run for all rects
                for rect in rects:

                    # update configs
                    dcfg.update(segmentation_method=segm_method,
                                covpp_method=cpp_method,
                                use_insc_ret=rect,
                                sequence_method='milp',
                                name_suffix=name_suffix,
                                start_cell_idx=0)

                    print(f"\n Running pipeline for method [{dcfg.method}] on map [{map_name}].")
                    # print(" --------------------------------------------------------------")
                    from covpp.main import run_pipeline
                    run_pipeline(map_name)

        print(" \n ====================================================== ")
        print(f"Finished processing {map_name}.")
        print(" ======================================================  \n")

    print("All maps processed successfully.")
