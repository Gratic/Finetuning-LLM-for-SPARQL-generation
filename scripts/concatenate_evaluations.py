import pandas as pd
import os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", required=True, type=str, help="The path to the folder containing evaluation.")
parser.add_argument("-o", "--output", required=True, type=str, help="The path to the output folder.")
parser.add_argument("-sn", "--save-name", required=True, type=str, help="The name of the file.")

args = parser.parse_args()

print(f"{os.listdir(args.folder)=}")

files = list(filter(lambda f: os.path.isfile(f), [os.path.join(args.folder, f) for f in os.listdir(args.folder)]))
print(files)

series = []
for f in files:
    serie = json.load(open(f, "r"))
    series.append(serie)

df = pd.DataFrame(data=series)

print(df[['model_name', 'num_rows', 'num_gen_fail']])
print(df[['num_exec_timeout', 'num_exec_fail', 'num_exec_empty', 'num_exec_to_eval']])
print(df[['num_eval', 'num_eval_empty']])
print(df[['bleu_score', 'meteor_score']])
print(df[['precision', 'recall', 'f1score']])
df.to_json(os.path.join(args.output, f"{args.save_name}.json"))