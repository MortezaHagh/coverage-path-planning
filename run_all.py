from main import run_pipeline


map_names = [
    "Map0",
    "Map1",
    "Map2",
    "Map4",
    "map_fr52",]

if __name__ == "__main__":
    for map_name in map_names:
        print(f"Running pipeline for {map_name}...")
        run_pipeline(map_name)
        print(f"Finished processing {map_name}.\n")
        print(" ######################################################## \n")
    print("All maps processed successfully.")
