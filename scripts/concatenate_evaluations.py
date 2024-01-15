import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", required=True, type=str, help="The path to the folder containing evaluation.")
parser.add_argument("-o", "--output", required=True, type=str, help="The path to the output folder.")
parser.add_argument("-sn", "--save-name", required=True, type=str, help="The name of the file.")

args = parser.parse_args()

files = [os.path.isfile(f) for f in os.listdir(args.folder)]

df = pd.DataFrame()

for f in files:
    serie = pd.read_json(f)
    df = pd.concat([df, serie])

df.to_json(os.path.join([args.output, f"{args.save_name}.json"]))