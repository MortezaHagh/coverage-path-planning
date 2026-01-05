""" class to create dataframe from json results files in MapT1 to MapT11 folders and then export to csv for paper analysis """

import os
import pandas as pd
import json

rename_planner = {
    'boust': 'BP',
    'contour': 'CB',
    'grid_energy_min': 'LEM'
}


class ResultsDataFrame:
    def __init__(self, base_path):
        self.base_path = base_path
        self.data = []

        self.segmentors = ['skeleton']
        self.planners = ['boust', 'contour', 'grid_energy_min']
        self.sequencers = ['dfs', 'milp']

    def load_results(self):
        for map_folder in os.listdir(self.base_path):

            # only MapT* folders
            if not map_folder.startswith('MapT'):
                continue
            map_path = os.path.join(self.base_path, map_folder)

            if os.path.isdir(map_path):
                for entry in os.scandir(map_path):
                    if entry.is_file() and entry.name.lower().endswith('.json'):
                        with open(entry.path, 'r') as f:
                            result_data = json.load(f)
                            result_data['map'] = map_folder
                            ful_method_name = os.path.splitext(entry.name)[0]
                            result_data['segmentor'] = self.get_segmentor_name(ful_method_name)
                            result_data['planner'] = self.get_planner_name(ful_method_name)
                            result_data['sequencer'] = self.get_sequencer_name(ful_method_name)
                            result_data['rect'] = 'rect' in ful_method_name
                            self.data.append(result_data)

    def get_segmentor_name(self, ful_method_name):
        for segmentor in self.segmentors:
            if segmentor in ful_method_name:
                if segmentor == 'skeleton':
                    return 'VoronoiBased'
                return segmentor
        return 'unknown'

    def get_planner_name(self, ful_method_name):
        for planner in self.planners:
            if planner in ful_method_name:
                return rename_planner.get(planner, 'unknown')
        return 'unknown'

    def get_sequencer_name(self, ful_method_name):
        for sequencer in self.sequencers:
            if sequencer in ful_method_name:
                return sequencer
        return 'unknown'

    def to_dataframe(self):
        df = pd.DataFrame(self.data)
        print(df.head())
        return df

    def export_to_csv(self, output_file):
        df = self.to_dataframe()
        df.to_csv(output_file, index=False)


if __name__ == "__main__":
    base_path = "./maps"  # Path to the directory containing MapT* folders
    output_file = "./results/coverage_results.csv"

    results_df = ResultsDataFrame(base_path)
    results_df.load_results()
    results_df.export_to_csv(output_file)
    print(f"Results exported to {output_file}")
