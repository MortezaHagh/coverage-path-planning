""" Run all coverage path planning methods on all maps. """
from covpp.base.configs import PipeLineConfig


# map_names = [
#     "Map4",
#     'MapPaper',
#     'MapPaper2',
#     "Map5",
#     'Map_office_a',
# ]
# map_names = ["Map_fr52", "Map4"]
map_names = ["MapT" + str(i) for i in range(1, 12)]
# map_names = ["MapPaper2"]
# map_names = ["Map1", "Map5"]

if __name__ == "__main__":

    # seg_methods = ["skeleton", "voronoi"]
    seg_methods = ["skeleton_e2"]
    # covpp_methods = ["boust", "contour", "grid_energy_min"]
    covpp_methods = ["contour"]
    # rects = [False, True]
    rects = [False]

    # covpp_method = ["boust"]

    dcfg = PipeLineConfig()

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
                    dcfg.update(segmentation_method=segm_method, covpp_method=cpp_method,
                                use_insc_ret=rect, sequence_method='milp', start_cell_idx=0)  # milp, dfs

                    print(f"\n Running pipeline for method [{dcfg.method}] on map [{map_name}].")
                    print(" --------------------------------------------------------------")
                    from covpp.main import run_pipeline
                    run_pipeline(map_name)
                    print(" ************************************************************************************ ")
                    print(" ************************************************************************************ ")
                    print(" ************************************************************************************ \n\n ")

        print(" \n ################################################################################################ ")
        print(" ################################################################################################ ")
        print(f"  Finished processing {map_name}.")
        print(" ################################################################################################")
        print(" ################################################################################################ \n\n")

    print("All maps processed successfully.")
