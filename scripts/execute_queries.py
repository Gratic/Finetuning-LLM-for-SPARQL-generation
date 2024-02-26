import sys
from pathlib import Path
sys.path.append(Path("modules").absolute().__str__())

from libwikidatallm.EntityFinder import WikidataAPI
import pandas as pd
from requests.exceptions import HTTPError, Timeout
import time
import argparse
import os
import re
from constants import PREFIX_TO_URL
from data_utils import load_dataset
from execution_utils import is_query_empty, can_add_limit_clause, add_relevant_prefixes_to_query

def send_query_to_api(query, api, timeout_limit, num_try):
    response = None
    while num_try > 0 and response == None and not is_query_empty(query):
        try:
            print(f"| Calling API... ", end="", flush=True)
            sparql_response = api.execute_sparql(query, timeout=timeout_limit)
            response = sparql_response.bindings if sparql_response.success else sparql_response.data
                
        except HTTPError as inst:
            if inst.response.status_code == 429:
                retry_after = int(inst.response.headers['retry-after'])
                print(f"| Retry-after: {retry_after} ", end="", flush=True)
                time.sleep(retry_after + 1)
                num_try -= 1
            else:
                print(f"| Exception occured ", end="", flush=True)
                response = "exception: " + str(inst) + "\n" + inst.response.text
        except Timeout:
            response = "timeout"
            print(f"| Response Timeout ", end="", flush=True)
        except Exception as inst:
            print(f"| Exception occured ", end="", flush=True)
            response = "exception: " + str(inst)
    return response if response != None else "exception: too many retry-after"

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

    api = WikidataAPI()
    
    num_processed = 0
    for (i, query) in df_dataset[args.column_name].items():
        print(f"row {str(num_processed)}/{len(df_dataset)-1} ".ljust(15), end="", flush=True)
        response = None
        
        if is_query_empty(query):
            response = "exception: query is empty"
            print(f"| Query is empty ", end="", flush=True)
        else:
            query = add_relevant_prefixes_to_query(query)
            
            if do_add_limit and can_add_limit_clause(query):
                query += f"\nLIMIT {answer_limit}"
            
            
            response = send_query_to_api(query=query,
                                    api=api,
                                    timeout_limit=timeout_limit,
                                    num_try=3)
        print(f"| done.", flush=True)
        
        df_dataset.at[i, 'execution'] = str(response)
        df_dataset.at[i, 'executed_query'] = query
        num_processed += 1

    output_file_path = os.path.join(args.output, f"{args.save_name}.parquet.gzip")
    try:
        df_dataset.to_parquet(output_file_path, engine="fastparquet", compression="gzip")
    except:
        df_dataset.to_parquet(output_file_path, compression="gzip")
        