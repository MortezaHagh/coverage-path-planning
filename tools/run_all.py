from covpp.base.configs import PipeLineConfig


map_names = [
    "Map0",
    "Map1",
    "Map2",
    "Map4",
    "map_fr52",]

# map_names = ["map_fr52", "Map4"]
# map_names = ["office_a"]

if __name__ == "__main__":

    covpp_method = ["boust", "contour", "grid_energy_min"]
    dcfg = PipeLineConfig()

    # run for all maps
    for map_name in map_names:

        # set start cell idx based on map
        if map_name == "map_fr52":
            S_CELL_IDX = 3
        else:
            S_CELL_IDX = 0

        # run for all segmentation methods
        for segm_method in ['voronoi']:

            # run for all coverage path planning methods
            for cpp_method in covpp_method:

                # use_insc_ret
                rects = [False]
                if segm_method == "voronoi":
                    rects = [True, False]
                    # rects = [True]

                # run for all rects
                for rect in rects:

                    # update configs
                    dcfg.update(segmentation_method=segm_method, covpp_method=cpp_method,
                                use_insc_ret=rect, sequence_method='milp', start_cell_idx=S_CELL_IDX)

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
