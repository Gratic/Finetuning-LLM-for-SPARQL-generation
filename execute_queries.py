from libwikidatallm.EntityFinder import WikidataAPI
import pandas as pd
import json
from requests.exceptions import HTTPError, Timeout
import time

dataset_path = "./datasets/final_queries_v1.1.json"
        
cleaned_queries = None
with open(dataset_path, 'r') as f:
    cleaned_queries = json.load(f)
    

df_dataset = pd.DataFrame(cleaned_queries)
df_dataset['execution'] = df_dataset.apply(lambda x: None, axis=1)
api = WikidataAPI()

for (i, row) in df_dataset['query'].items():
    print(f"row {str(i)}/{len(df_dataset)} ".ljust(15), end="", flush=True)
    response = None
    
    num_try_left = 2
    
    while num_try_left > 0 and response == None:
        try:
            print(f"| Calling API... ", end="", flush=True)
            sparql_response = api.execute_sparql(row, timeout=30)
            response = sparql_response.bindings if sparql_response.success else sparql_response.data
            
        except HTTPError as inst:
            if inst.response.status_code == 429:
                retry_after = int(inst.response.headers['retry-after'])
                print(f"| Retry-after: {retry_after} ", end="", flush=True)
                time.sleep(retry_after)
                num_try_left -= 1
            else:
                print(f"| Exception occured ", end="", flush=True)
                num_try_left = 0
                response = "exception: " + str(inst)
        except Timeout:
            response = "timeout"
            num_try_left = 0
            print(f"| Response Timeout ", end="", flush=True)
        except Exception as inst:
            print(f"| Exception occured ", end="", flush=True)
            response = "exception: " + str(inst)
            num_try_left = 0
    print(f"| done.", flush=True)
    
    df_dataset.at[i, 'execution'] = response

with open("./outputs/queries_with_execution_results.json", "w") as f:
    data_json = df_dataset.to_json()
    f.write(data_json)