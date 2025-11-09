from pathlib import Path
import json
import pandas as pd


def collect_results(parameter_path):
    """Collect all result.json files and extract relevant data."""
    data = []

    # Find all result.json files recursively
    for json_file in Path('.').rglob('result.json'):
        try:
            with open(json_file, 'r') as f:
                result = json.load(f)

            # Extract required fields
            map_name = result.get('map', 'Unknown')
            method = result.get('method', 'Unknown')

            # Navigate through nested dictionary using parameter_path
            value = result
            for key in parameter_path:
                value = value.get(key, {})

            # If final value is not a dict, it's our target value
            if not isinstance(value, dict) and value is not None:
                data.append({
                    'map': map_name,
                    'method': method,
                    'value': value
                })

        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            print(f"Error reading {json_file}: {e}")

    return data


def create_table(data):
    """Create a pivot table with methods as rows and maps as columns."""
    if not data:
        print("No valid data found.")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Create pivot table
    table = df.pivot_table(
        index='method',
        columns='map',
        values='value',
        aggfunc='first'  # In case of duplicates, take first value
    )

    return table


def main():
    # Define available parameters
    parameters = {
        'n_paths': ['planner_stats', 'n_paths'],
        'len_m': ['planner_stats', 'len_m'],
        'yaw_changes': ['planner_stats', 'yaw_changes'],
        'coverage_p': ['zone_stats', 'coverage_p'],
        'overlap_p': ['zone_stats', 'overlap_p']
    }

    # User selection (you can modify this part)
    print("Available parameters:")
    for i, param in enumerate(parameters.keys(), 1):
        print(f"{i}. {param}")

    # Option 1: Interactive selection
    choice = input("\nEnter parameter name (or number): ").strip()

    # Handle numeric choice
    if choice.isdigit():
        param_list = list(parameters.keys())
        if 1 <= int(choice) <= len(param_list):
            selected_param = param_list[int(choice) - 1]
        else:
            print("Invalid choice, using default 'len_m'")
            selected_param = 'len_m'
    else:
        selected_param = choice if choice in parameters else 'len_m'

    # Option 2: Direct assignment (uncomment to use)
    # selected_param = 'coverage_p'  # Change this to desired parameter

    parameter_path = parameters.get(selected_param, parameters['len_m'])

    # Collect data from all result.json files
    data = collect_results(parameter_path)

    # Create table
    table = create_table(data)

    if table is not None:
        print(f"\n{selected_param.upper()} by Method and Map:")
        print("=" * 60)
        print(table)

        # Generate filename based on parameter
        dir_data = "/home/morteza/Documents/00-CPP/00-CPP/results/All"
        filename = dir_data + f'results_{selected_param}.csv'

        # Optional: Save to CSV
        table.to_csv(filename)
        print(f"\nTable saved to '{filename}'")

        # Display some statistics
        print("\nSummary:")
        print(f"Parameter: {selected_param}")
        print(f"Methods found: {len(table.index)}")
        print(f"Maps found: {len(table.columns)}")
        print(f"Total entries: {len(data)}")


if __name__ == "__main__":
    main()
