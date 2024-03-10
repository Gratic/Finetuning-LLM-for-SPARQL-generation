import sys
from pathlib import Path
sys.path.append(Path("modules").absolute().__str__())

from evaluation_utils import process_dataset_for_evaluation
import argparse
import json
import logging
import os

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

    df_gold,df_gold_exec_timeout,df_gold_exec_fail,df_gold_exec_empty,df_gold_exec_to_eval,df_gold_eval = process_dataset_for_evaluation(args.gold, prefix="gold_")

    data = {
        "df_gold": df_gold.to_json(),
        "df_gold_exec_timeout": df_gold_exec_timeout.to_json(),
        "df_gold_exec_fail": df_gold_exec_fail.to_json(),
        "df_gold_exec_empty": df_gold_exec_empty.to_json(),
        "df_gold_exec_to_eval": df_gold_exec_to_eval.to_json(),
        "df_gold_eval": df_gold_eval.to_json(),
    }
    
    save_path = os.path.join(args.output, f"{args.save_name}.json")
    
    with open(save_path, "w") as f:
        f.write(json.dumps(data))