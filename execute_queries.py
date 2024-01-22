from libwikidatallm.EntityFinder import WikidataAPI
import pandas as pd
from requests.exceptions import HTTPError, Timeout
import time
import argparse
import os

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

    df_dataset = pd.read_parquet(dataset_path, engine="fastparquet")
    df_dataset['execution'] = df_dataset.apply(lambda x: None, axis=1)

    api = WikidataAPI()
    
    num_processed = 0
    for (i, query) in df_dataset[args.column_name].items():
        print(f"row {str(num_processed)}/{len(df_dataset)} ".ljust(15), end="", flush=True)
        response = None
        is_empty = False
        
        if query is None or query == "" or len(query) == 0:
            response = ""
            is_empty = True
            print(f"| Query is empty ", end="", flush=True)
            
        
        num_try_left = 3
        
        if not is_empty and do_add_limit and "COUNT" in query and not "LIMIT" in query:
            query += f"\nLIMIT {answer_limit}"
        
        while num_try_left > 0 and response == None and not is_empty:
            try:
                print(f"| Calling API... ", end="", flush=True)
                sparql_response = api.execute_sparql(query, timeout=timeout_limit)
                response = sparql_response.bindings if sparql_response.success else sparql_response.data
                
            except HTTPError as inst:
                if inst.response.status_code == 429:
                    retry_after = int(inst.response.headers['retry-after'])
                    print(f"| Retry-after: {retry_after} ", end="", flush=True)
                    time.sleep(retry_after + 1)
                    num_try_left -= 1
                else:
                    print(f"| Exception occured ", end="", flush=True)
                    num_try_left -= 1
                    response = "exception: " + str(inst)
            except Timeout:
                response = "timeout"
                num_try_left -= 1
                print(f"| Response Timeout ", end="", flush=True)
            except Exception as inst:
                print(f"| Exception occured ", end="", flush=True)
                response = "exception: " + str(inst)
                num_try_left -= 1
        print(f"| done.", flush=True)
        
        df_dataset.at[i, 'execution'] = str(response)
        num_processed += 1

    df_dataset.to_parquet(os.path.join(args.output, f"{args.save_name}.parquet.gzip"), engine="fastparquet", compression="gzip")