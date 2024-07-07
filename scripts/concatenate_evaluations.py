import pandas as pd
import os
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", required=True, type=str, help="The path to the folder containing evaluation.")
    parser.add_argument("-o", "--output", required=True, type=str, help="The path to the output folder.")
    parser.add_argument("-sn", "--save-name", required=True, type=str, help="The name of the file.")

    return parser.parse_args()

def main(args: argparse.Namespace):
    os.makedirs(args.output, exist_ok=True)

    # List all JSON files in the input folder
    json_files = [f for f in os.listdir(args.folder) if f.endswith('.json')]

    # Read and stack all JSON files
    series = []
    for ff in json_files:
        file_path = os.path.join(args.folder, ff)
        with open(file_path, 'r') as f:
            data = json.load(f)
        series.append(data)

    # Stack all dataframes
    if series:
        final_df = pd.DataFrame(series)

        # Save the final dataframe as JSON
        output_path = os.path.join(args.output, f"{args.save_name}.json")
        final_df.to_json(output_path)
        print(f"Stacked JSON saved to {output_path}")
    else:
        print("No JSON files found in the specified folder.")
        
if __name__ == "__main__":
    args = parse_args()
    main(args)