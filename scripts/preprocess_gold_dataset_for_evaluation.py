from typing import Dict, Union
import argparse
import logging
import os
import pandas as pd
from ast import literal_eval
import json

def failed_generation_index(dataset: pd.DataFrame):
    return dataset.loc[dataset['has_error'] == True].index


def safe_eval(execution: str):
    """Evaluates """
    try:
        return literal_eval(execution)
    except Exception as inst:
        logging.error(f"Exception occured while evaluating: {inst}.")
        print(f"Exception occured while evaluating: {inst}.")
        return None

def eval_dataset(dataset: pd.DataFrame, col_name: str = "eval"):
    df_eval = dataset.copy()
    df_eval[col_name] = df_eval.apply(lambda x: safe_eval(x['execution']), axis=1)
    return df_eval[~df_eval[col_name].isnull()]

def get_nested_values(element: Union[Dict, str, None]):
    values = []
    if isinstance(element, dict):
        for k, v in element.items():
            if isinstance(v, dict):
                values += get_nested_values(v)
            elif isinstance(v, str):
                if 'value' in k:
                    values.append(v)
    elif isinstance(element, list):
        for el in element:
            values += get_nested_values(el)
    elif element is None:
        values = []
    else:
        logging.error(f"get_nested_values doesn't have an implementation for: {type(element)}.")
        raise TypeError(f"Compatible types are Dict and List, found: {type(element)}.")
    return values

def load_dataset(path: str):
    if path.endswith(('.parquet', '.parquet.gzip')):
        return pd.read_parquet(path, engine='auto')
    elif path.endswith('.json'):
        return pd.read_json(path)
    
def safe_loc(x, df, column, default=None):
    try:
        ans = df[[column]].loc[str(x.name)]
    except:
        ans = default
    return ans

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Preprocess Gold Dataset",
                                    description="Preprocess the gold dataset for evaluation bench.")
    parser.add_argument('-g', '--gold', required=True, type=str, help="The path to the gold dataset (dataset with answers).")
    parser.add_argument('-o', '--output', required=True, type=str, help="Folder to output the results.")
    parser.add_argument('-sn', '--save-name', required=True, type=str, help="Name of the save file.")
    parser.add_argument("-log", "--log-level", type=str, help="Logging level (debug, info, warning, error, critical).", default="warning")
    parser.add_argument("-logf", "--log-file", type=str, help="Logging file.", default="")

    args = parser.parse_args()
    
    numeric_log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_log_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}.")
    logging.basicConfig(filename=args.log_file if args.log_file else None, level=numeric_log_level)

    if not os.path.exists(args.gold):
        raise FileNotFoundError(f"The gold dataset file not found with path: {args.gold}")

    df_gold = load_dataset(args.gold)
    
    df_gold_exec_timeout = df_gold.loc[df_gold['execution'] == 'timeout']
    df_gold_exec_fail = df_gold.loc[df_gold['execution'].str.startswith('exception')]
    df_gold_exec_empty = df_gold.loc[df_gold['execution'].isnull()]
    df_gold_exec_to_eval = df_gold.drop(df_gold_exec_timeout.index).drop(df_gold_exec_fail.index).drop(df_gold_exec_empty.index)
    df_gold_eval = eval_dataset(df_gold_exec_to_eval, "gold_eval")

    data = {
        "df_gold": df_gold.to_json(),
        "df_gold_exec_timeout": df_gold_exec_timeout.to_json(),
        "df_gold_exec_fail": df_gold_exec_fail.to_json(),
        "df_gold_exec_empty": df_gold_exec_empty.to_json(),
        "df_gold_exec_to_eval": df_gold_exec_to_eval.to_json(),
        "df_gold_eval": df_gold_eval.to_json()
    }
    
    save_path = os.path.join(args.output, f"{args.save_name}.json")
    
    with open(save_path, "w") as f:
        f.write(json.dumps(data))