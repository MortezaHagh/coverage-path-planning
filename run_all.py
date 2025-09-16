from scripts.base.configs import DistrictConfigs


map_names = [
    "Map0",
    "Map1",
    "Map2",
    "Map4",
    "map_fr52",]

if __name__ == "__main__":

    covpp_method = ["m2_boust", "m3_contour"]
    dcfgs = DistrictConfigs()

    # run for all maps
    for map_name in map_names:

        # run for all methods
        for cpp_method in covpp_method:

            # update for m3_contour
            rects = [False]
            if cpp_method == "m3_contour":
                rects = [True, False]

            # run for all rects
            for rect in rects:

                # update configs
                dcfgs.update(covpp_method=cpp_method, use_insc_ret=rect)

                print(" ######################################################## \n")
                print(f"Running pipeline for method {dcfgs.method} on map {map_name} ... ")
                print(" ######################################################## \n")
                from main import run_pipeline
                run_pipeline(map_name)
                print(f"Finished processing {map_name}.\n")
                print(" ######################################################## \n")

    print("All maps processed successfully.")
