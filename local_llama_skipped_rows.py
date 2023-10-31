import http.client
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import re
import datetime
import os

# Connections options
SERVER_ADDR = "127.0.0.1"
SERVER_PORT = "8080"
SERVER_COMPLETION_ENDPOINT = "/completion"

POST_COMPLETION_HEADERS = {"Content-Type":"application/json"}
 
# Dataset processing options
SKIPPED_ROWS_PATH  = "./outputs/generated_prompts/20231009-1442_skipped_rows.json"
CLEANED_QUERIES_PATH = "./datasets/cleaned_queries/cleaned_queries.json"

SYSTEM_PROMPT = "<<SYS>>This is a conversation between User and Llama, a friendly chatbot. Llama is helpful, kind, honest, good at writing, and never fails to answer any requests immediately and with precision.<</SYS>>\n"
MAX_NUMBER_OF_TRY_PER_PROMPT = 25
RETRY_IF_ANSWER_CONTAINS = ["SELECT", "GROUP"]


# Dataset output options
OUTPUT_PATH = "./outputs/generated_prompts/"

# Prompt processing/Generation options
NUMBER_OF_TOKEN_TO_PREDICT = 512
TEMPERATURE = 0.4 # (default = 0.8)

with open(CLEANED_QUERIES_PATH, 'r') as f:
  data_cleaned_queries = json.load(f)

with open(SKIPPED_ROWS_PATH, 'r') as f:
  data_skipped_rows = json.load(f)
  
def row_into_string(row):
  return f"QUERY=\"{row['query']}\" DESCRIPTION=\"{row['description']}\" CONTEXT=\"{row['context']}\""

def init_prompt(sys_prompt, query):
  return f"""<s>[INST] {sys_prompt}
{row_into_string(query)}

User: Read QUERY, DESCRIPTION and CONTEXT. There is a machine capable of writing the given QUERY if we ask it the right prompt. Please do not include parts of QUERY in your answers. Give a list of 3 prompts that would give QUERY. [\INST]
Llama:"""

def prepare_request_payload(prompt):
    data_dict = dict()
    data_dict["prompt"] = prompt
    data_dict["n_predict"] = NUMBER_OF_TOKEN_TO_PREDICT
    data_dict["temperature"] = TEMPERATURE

    data_json = json.dumps(data_dict)
    return data_json

def get_completion_response(body_json):
    conn = http.client.HTTPConnection(f"{SERVER_ADDR}:{SERVER_PORT}")
    conn.request(method="POST",
             url=SERVER_COMPLETION_ENDPOINT,
             headers=POST_COMPLETION_HEADERS, 
             body=body_json)

    response = conn.getresponse()

    if response.status != 200:
        print(response.status, response.reason)
        exit(1)

    answer = response.read()
    answer_dict = json.loads(answer)
    return answer_dict

def are_results_acceptable(results):
    is_good_quality = True
    
    if len(results) != 3:
        return False
    
    for result in results:
        for word in RETRY_IF_ANSWER_CONTAINS:
            if word in result:
                is_good_quality = False
    return is_good_quality
                

df = pd.DataFrame(data_cleaned_queries)
df = pd.concat([df.drop(['metadata'], axis=1), df['metadata'].apply(pd.Series)], axis=1)
df["prompt"] = df.apply(lambda x: init_prompt(SYSTEM_PROMPT, x), axis=1)
df["result"] = df.apply(lambda: "", axis=1)

skipped_rows = list(data_skipped_rows)

df = df.iloc[skipped_rows]

compiled_regex_pattern = re.compile(r'\"[A-Z].*\"', flags=0)

prompts = df["prompt"]
skipped_rows = []

for row_index in tqdm(range(0, len(df))):
    number_of_try_left = MAX_NUMBER_OF_TRY_PER_PROMPT
    found_results = False
    
    while number_of_try_left > 0 and not found_results:
        prompt = prompts.iat[row_index]
        
        data_json = prepare_request_payload(prompt)
        answer = get_completion_response(data_json)
        
        result = compiled_regex_pattern.findall(answer['content'])
        
        if not are_results_acceptable(result):
            number_of_try_left -= 1
            continue
        
        found_results = True
        df['result'].iat[row_index] = result
    
    if not found_results:
        print(f"No results found for: {df.iloc[row_index].name}")
        skipped_rows.append(df.iloc[row_index].name)
    
# for row_number in range(STARTING_ROW_OFFSET,range_stop):
#     print(df['result'].iat[row_number])
    
dataframe_json_dump = df.iloc[0:len(df)].to_json()

os.makedirs(OUTPUT_PATH, exist_ok=True)
with open(f"{OUTPUT_PATH}{datetime.datetime.now().strftime('%Y%m%d-%H%M')}_skipped_rows_generated_prompts.json", "w") as outfile:
    outfile.write(dataframe_json_dump)

if len(skipped_rows) > 0:
    skipped_rows_json_dump = json.dumps(skipped_rows)
    with open(f"{OUTPUT_PATH}{datetime.datetime.now().strftime('%Y%m%d-%H%M')}_skipped_rows.json", "w") as outfile:
        outfile.write(skipped_rows_json_dump)
