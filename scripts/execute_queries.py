import sys
from pathlib import Path
sys.path.append(Path("modules").absolute().__str__())

import argparse
import os
from data_utils import load_dataset
from execution_utils import prepare_and_send_query_to_api

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="SPARQL Queries Executor",
                                    description="Execute queries on Wikidata's SPARQL endpoint")
    parser.add_argument('-d', '--dataset', required=True, type=str, help="Path to the dataset parquet file.")
    parser.add_argument('-c', '--column-name', type=str, help="The column where the queries to be executed are.", default="query")
    parser.add_argument('-t', '--timeout', type=int, help="The amount of time for the HTTP client to wait before timeout.", default=60)
    parser.add_argument('-l', "--limit", type=int, help="If the limit is >0 a LIMIT clause will be added to all non COUNT queries.", default=0)
    parser.add_argument('-o', "--output", required=True, type=str, help="Path to the directory the save file will be.")
    parser.add_argument('-sn', "--save-name", required=True, type=str, help="Name of the save file.")

    args = parser.parse_args()

    dataset_path = args.dataset
    answer_limit = args.limit
    timeout_limit = args.timeout
    do_add_limit = answer_limit > 0

    df_dataset = load_dataset(dataset_path)
    
    if not args.column_name in df_dataset.columns:
        raise ValueError(f"The column '{args.column_name}' was not found in the dataset. Columns are {', '.join(df_dataset.columns)}.")
    
    df_dataset['execution'] = df_dataset.apply(lambda x: None, axis=1)
    df_dataset['executed_query'] = df_dataset.apply(lambda x: None, axis=1)
    
    num_processed = 0
    for (i, query) in df_dataset[args.column_name].items():
        query, response = prepare_and_send_query_to_api(
            query=query,
            index=num_processed,
            num_of_rows=len(df_dataset)-1,
            answer_limit=answer_limit,
            timeout_limit=timeout_limit,
            do_add_limit=do_add_limit,
            do_print=True
        )
        
        df_dataset.at[i, 'execution'] = str(response)
        df_dataset.at[i, 'executed_query'] = query
        num_processed += 1

    output_file_path = os.path.join(args.output, f"{args.save_name}.parquet.gzip")
    try:
        df_dataset.to_parquet(output_file_path, engine="fastparquet", compression="gzip")
    except:
        df_dataset.to_parquet(output_file_path, compression="gzip")
        