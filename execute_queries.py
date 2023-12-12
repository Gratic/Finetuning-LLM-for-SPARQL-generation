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
    print(f"row {str(i)}/{len(df_dataset)} ".ljust(15), end="")
    response = None
    
    num_try_left = 2
    
    while num_try_left > 0 and response == None:
        try:
            print(f"| Calling API... ", end="")
            response = api.execute_sparql(row, timeout=30)
        except HTTPError as inst:
            if inst.response.status_code == 429:
                retry_after = inst.response.headers['retry-after']
                print(f"| Retry-after: {retry_after} ", end="")
                time.sleep(retry_after)
                num_try_left -= 1
        except Timeout:
            response = "timeout"
            num_try_left = 0
            print(f"| Response Timeout ", end="")
    print(f"| done.")
    
    df_dataset.at[i, 'execution'] = response

with open("./outputs/queries_with_execution_results.json", "w") as f:
    data_json = df_dataset.to_json()
    f.write(data_json)