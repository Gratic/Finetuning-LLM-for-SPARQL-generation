import sys
from pathlib import Path
sys.path.append(Path("modules").absolute().__str__())

import argparse
import logging
import os
import json
from data_utils import eval_dataset, get_nested_values, load_dataset, make_dataframe_from_sparql_response
from evaluation_utils import keep_id_columns

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
    df_gold_eval['gold_get_nested_values'] = df_gold_eval.apply(lambda x: get_nested_values(x['gold_eval']), axis=1)
    df_gold_eval['gold_eval_df'] = df_gold_eval.apply(lambda x: make_dataframe_from_sparql_response(x['gold_eval']), axis=1)
    df_gold_eval['gold_id_columns'] = df_gold_eval.apply(lambda x: keep_id_columns(x['gold_eval_df']), axis=1)

    data = {
        "df_gold_eval": df_gold_eval.to_json()
    }
    
    save_path = os.path.join(args.output, f"{args.save_name}.json")
    
    with open(save_path, "w") as f:
        f.write(json.dumps(data))