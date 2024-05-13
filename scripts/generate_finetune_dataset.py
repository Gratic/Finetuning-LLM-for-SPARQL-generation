import sys
from pathlib import Path
sys.path.append(Path("modules").absolute().__str__())

import pandas as pd
import argparse
import re
from pathlib import Path
from data_utils import load_dataset, set_seed
from evaluation_utils import process_dataset_for_evaluation
import numpy as np
import json

def keep_working_queries(df):
    df_timeout = df.loc[df['execution'] == 'timeout']
    df_fail = df.loc[df['execution'].str.startswith('exception')]
    df_empty = df.drop(df_timeout.index).drop(df_fail.index).loc[df['execution'].str.startswith("[]")]
    df = df.drop(df_timeout.index).drop(df_fail.index).drop(df_empty.index)
    return df

def generate_gold_dataset_for_split(save_path: Path, split: pd.DataFrame):
    df_gold,df_gold_exec_timeout,df_gold_exec_fail,df_gold_exec_empty,df_gold_exec_to_eval,df_gold_eval = process_dataset_for_evaluation(split, prefix="gold_", execution_col="gold_execution")

    data = {
        "df_gold": df_gold.to_json(),
        "df_gold_exec_timeout": df_gold_exec_timeout.to_json(),
        "df_gold_exec_fail": df_gold_exec_fail.to_json(),
        "df_gold_exec_empty": df_gold_exec_empty.to_json(),
        "df_gold_exec_to_eval": df_gold_exec_to_eval.to_json(),
        "df_gold_eval": df_gold_eval.to_json(),
    }
    
    save_path.write_text(json.dumps(data))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Path to the pandas dataset with result column and query column.", required=True)
    parser.add_argument("-kw", "--keep-working", action="store_true", help="Keep only working queries (deduced from execution).")
    parser.add_argument("-o", "--output", type=str, help="Path to output directory.", default="./outputs/")
    parser.add_argument("-sn", "--save-name", type=str, help="Save name, splits will be suffixed with _train, _test, _valid.", default="finetune_dataset")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode (show print).")
    parser.add_argument("-rand", "--random-seed", type=int, help="Set up a random seed if specified.", default=0)

    args = parser.parse_args()

    df = load_dataset(args.input)
    
    if args.random_seed == 0:
        set_seed(args.random_seed)

    if args.keep_working:
        df = keep_working_queries(df)

    # TODO: hard coded values... the config_dataset.ini to use final_queries_to_finetunning_dataset.py can mess with the column names.
    df['basic_input'] = df.apply(lambda x: x['basic_result'], axis=1)
    df['templated_input'] = df.apply(lambda x: x['templated_result'], axis=1)
    df['target_raw'] = df.apply(lambda x: x['query'], axis=1)
    df['target_template'] = df.apply(lambda x: x['query_templated'], axis=1)
    df['gold_execution'] = df.apply(lambda x: x['execution'], axis=1)
    df['gold_executed_query'] = df.apply(lambda x: x['executed_query'], axis=1)
    
    df_output = df[['basic_input', 'templated_input', 'target_raw', 'target_template', 'gold_execution', 'gold_executed_query']]
    
    if args.debug:
        print(f"{df_output.iloc[[0]]=}")
        print(f"{df_output.iloc[[0]]['basic_input']=}")
        print(f"{df_output.iloc[[0]]['templated_input']=}")
        print(f"{df_output.iloc[[0]]['target_raw']=}")
        print(f"{df_output.iloc[[0]]['target_template']=}")
        print(f"{df_output.iloc[[0]]['gold_execution']=}")
        print(f"{df_output.iloc[[0]]['gold_executed_query']=}")
    
    # Split the data into train, validation and test sets in a stratified manner using numpy's random permutation function. 75% of data will be used for training, remaining 5% will be used for validation and remaining 20% for testing.
    df_train, df_valid, df_test = np.split(df_output.sample(frac=1, random_state=args.random_seed), [int(.75*len(df_output)), int(.80*len(df_output))]) 
    
    if args.debug:
        print(f"{len(df_output)=}")
        print(f"{len(df_train)=}")
        print(f"{len(df_valid)=}")
        print(f"{len(df_test)=}")
    
    folder = Path(args.output)
    
    train_file = str(folder / f"{args.save_name}_train.pkl")
    print(f"train dataset saved at: {train_file}") 
    df_train.to_pickle(train_file)
    
    valid_file = str(folder / f"{args.save_name}_valid.pkl")
    print(f"valid dataset saved at: {valid_file}")
    df_valid.to_pickle(valid_file)
    
    test_file = str(folder / f"{args.save_name}_test.pkl")
    print(f"test dataset saved at: {test_file}")
    df_test.to_pickle(test_file)
    
    generate_gold_dataset_for_split(save_path=folder / f"gold_{args.save_name}_train.json", split=df_train)
    print("Gold dataset for train saved")
    
    generate_gold_dataset_for_split(save_path=folder / f"gold_{args.save_name}_valid.json", split=df_valid)
    print("Gold dataset for valid saved")
    
    generate_gold_dataset_for_split(save_path=folder / f"gold_{args.save_name}_test.json", split=df_test)
    print("Gold dataset for test saved")