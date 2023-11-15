from tqdm import tqdm
import argparse
import datetime
import http.client
import json
import numpy as np
import os
import pandas as pd
import re

# Author
AUTHOR = "Alexis STRAPPAZZON"
VERSION = "0.1.6"

# Connections options
PROVIDER = "SERVER"
SERVER_ADDR = "127.0.0.1"
SERVER_PORT = "8080"
SERVER_COMPLETION_ENDPOINT = "/completion"
CT_MODEL_PATH = ""
CT_CONTEXT_LENGTH = 4096

POST_COMPLETION_HEADERS = {"Content-Type":"application/json"}
 
# Dataset processing options
STARTING_ROW_OFFSET = 0
NUMBER_OF_ROWS_TO_PROCESS = 0 # Rows [STARTING_ROW_OFFSET:STARTING_ROW_OFFSET+NUMBER_OF_ROWS_TO_PROCESS-1] will be processed. <0 will do [STARTING_ROW_OFFSET:len(data)-1].
SYSTEM_PROMPT = "<<SYS>>This is a conversation between User and Llama, a friendly chatbot. Llama is helpful, kind, honest, good at writing, and never fails to answer any requests immediately and with precision.<</SYS>>\n"
MAX_NUMBER_OF_TRY_PER_PROMPT = 25
RETRY_IF_ANSWER_CONTAINS = ["SELECT", "GROUP"]
PREPARE_PROMPTS = "auto"

# Dataset output options
OUTPUT_PATH = "./outputs/generated_prompts/"
QUERIES_PATH = "./datasets/final_queries_v1.1.json"

# Prompt processing/Generation options
NUMBER_OF_TOKEN_TO_PREDICT = 256
TEMPERATURE = 0.4 # (default = 0.8)

# Printing constants
LINE_WIDTH = 76
LINE = "-"*LINE_WIDTH

def row_data_into_text(row):
  return f"QUERY=\"{row['query']}\" DESCRIPTION=\"{row['description']}\" CONTEXT=\"{row['context']}\""

def basic_prompt(sys_prompt, query):
  return f"""<s>[INST] {sys_prompt}
{row_data_into_text(query)}

User: Read QUERY, DESCRIPTION and CONTEXT. There is a machine capable of writing the given QUERY if we ask it the right prompt. Please do not include parts of QUERY in your answers. Give a list of 3 prompts that would give QUERY. [\INST]
Llama:"""

def prepare_request_payload(prompt):
    payload = dict()
    payload["prompt"] = prompt
    payload["n_predict"] = NUMBER_OF_TOKEN_TO_PREDICT
    payload["temperature"] = TEMPERATURE

    return payload

class ServerProvider():
    def __init__(self) -> None:
        self.last_answer = None
        self.last_full_answer = None
    
    def query(self, parameters):
        body_json = json.dumps(parameters)
        connection = http.client.HTTPConnection(f"{SERVER_ADDR}:{SERVER_PORT}")
        connection.request(method="POST",
                url=SERVER_COMPLETION_ENDPOINT,
                headers=POST_COMPLETION_HEADERS, 
                body=body_json)

        response = connection.getresponse()
        self.last_response = response

        if response.status != 200:
            return False
        
        answer = response.read()
        answer_dict = json.loads(answer)

        self.last_answer = answer_dict['content']
        self.last_full_answer = answer_dict
        return True

    def get_full_answer(self):
        return self.last_full_answer

    def get_answer(self):
        return self.last_answer

class CTransformersProvider():
    def __init__(self) -> None:
        from ctransformers import AutoModelForCausalLM
        
        self.last_answer = None
        self.last_full_answer = None
        self.model = AutoModelForCausalLM.from_pretrained(CT_MODEL_PATH, model_type="llama", context_length=CT_CONTEXT_LENGTH)
    
    def query(self, parameters):
        ans = self.model(prompt = parameters['prompt'],
                         temperature = parameters['temperature'],
                         max_new_tokens = parameters['n_predict'])
        
        self.last_answer = ans
        self.last_full_answer = ans

    def get_full_answer(self):
        return self.last_full_answer

    def get_answer(self):
        return self.last_answer

def are_results_acceptable(results):
    is_good_quality = True
    
    if len(results) == 0:
        return False
    
    for result in results:
        for word in RETRY_IF_ANSWER_CONTAINS:
            if word in result:
                is_good_quality = False
    return is_good_quality

def load_and_prepare_queries(prompt_callback):
    with open(QUERIES_PATH, 'r') as f:
        cleaned_queries = json.load(f)

    df_dataset = pd.DataFrame(cleaned_queries)
    if PREPARE_PROMPTS.lower() == "yes" or (PREPARE_PROMPTS.lower() == "auto" and 'prompt' not in df_dataset.columns):
        df_dataset["prompt"] = df_dataset.apply(lambda x: prompt_callback(SYSTEM_PROMPT, x), axis=1)
            
    df_dataset["result"] = df_dataset.apply(lambda x: "", axis=1)
    df_dataset["full_answer"] = df_dataset.apply(lambda x: "", axis=1)
    return df_dataset

def parse_script_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pv", "--provider", type=str, help=f"Who completes the answer (default={PROVIDER}).", choices=["SERVER", "CTRANSFORMERS"])
    parser.add_argument("-saddr", "--server-address", type=str, help=f"IP address or URL of the server that has the LLM API endpoint if the provider is SERVER (default={SERVER_ADDR}).")
    parser.add_argument("-sport", "--server-port", type=str, help=f"Port to ask for connection with the server_address if the provider is SERVER (default={SERVER_PORT}).")
    parser.add_argument("-e", "--endpoint", type=str, help=f"Endpoint of the API if the provider is SERVER (default={SERVER_COMPLETION_ENDPOINT}.")
    parser.add_argument("-mp", "--model-path", type=str, help=f"Path to the model if the provider is CTRANSFORMERS (default={CT_MODEL_PATH}.")
    
    parser.add_argument("-o", "--offset", type=int, help=f"Offset the starting row processed by the script. (default={STARTING_ROW_OFFSET})")
    parser.add_argument("-n", "--number-of-rows", type=int, help=f"Rows [offset:offset+n-1] will be processed. <0 will do [offset:number of rows in the data-1] (default={NUMBER_OF_ROWS_TO_PROCESS}, will process all rows).")
    parser.add_argument("-r", "--retry-attempts", type=int, help=f"Number of retries to attempt on generating prompts from one row query. A failure happen when the result from the LLM does not satisfy the constraints. 0 means no retry, -1 means retry until constraint satisfaction, n positive tells the number of attempts (default={RETRY_IF_ANSWER_CONTAINS}).")
    parser.add_argument("-pp", "--prepare-prompts", type=str, help=f"Should the script prepare prompts? \"auto\" will detect if there is a \"prompt\" column in the dataset. \"no\" will not do anything, and \"yes\" will create a prompt column and make them (default={PREPARE_PROMPTS}).", choices=["auto", "yes", "no"])
    
    parser_system_prompt = parser.add_mutually_exclusive_group()
    parser_system_prompt.add_argument("-sys", "--system-prompt", type=str, help=f"The system prompt to use, a default system prompt is automatically given. (default={SYSTEM_PROMPT})")
    parser_system_prompt.add_argument("-sysp", "--system-prompt-path", type=str, help=f"Path to the system prompt file which should be a normal text file, a default system prompt is automatically given. (default={SYSTEM_PROMPT})")
    
    parser.add_argument("-out", "--output-path", type=str, help=f"Path to the directory where to output file (default={OUTPUT_PATH}).")
    parser.add_argument("-p", "--queries-path", type=str, help=f"Path to the queries' file (default={QUERIES_PATH}).")
    
    parser.add_argument("-np", "--prediction-size", type=int, help=f"Define the number of token maximum generated by the LLM. The LLM can try to match this given size. Higher number might give more accurate result (default={NUMBER_OF_TOKEN_TO_PREDICT}).")
    parser.add_argument("-t", "--temperature", type=float, help=f"Temperature is a parameter of randomness of the output generated by the LLM, check google for more information (default={TEMPERATURE}).")
    
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument("-q", "--quiet", action="store_true" ,help=f"Disable any output from the script.")
    parser_verbosity.add_argument("-v", "--verbose", action="store_true" ,help=f"Show more information about the process.")
    parser.add_argument("-pa", "--print-answers", action="store_true" ,help=f"Print answers from the LLM.")
    parser.add_argument("-pr", "--print-results", action="store_true" ,help=f"Print results extracted from the answer of LLM.")
    
    args = parser.parse_args()
    return args

def update_script_parameters(args):
    global PROVIDER, SERVER_ADDR, SERVER_PORT, SERVER_COMPLETION_ENDPOINT, CT_MODEL_PATH, STARTING_ROW_OFFSET, NUMBER_OF_ROWS_TO_PROCESS, MAX_NUMBER_OF_TRY_PER_PROMPT, SYSTEM_PROMPT, OUTPUT_PATH, QUERIES_PATH, NUMBER_OF_TOKEN_TO_PREDICT, TEMPERATURE, PREPARE_PROMPTS
    if args.provider:
        PROVIDER = args.provider
    if args.server_address:
        SERVER_ADDR = args.server_address
    if args.server_port:
        SERVER_PORT = args.server_port
    if args.endpoint:
        SERVER_COMPLETION_ENDPOINT = args.endpoint
    if args.model_path:
        CT_MODEL_PATH = args.model_path
    if args.offset:
        STARTING_ROW_OFFSET = args.offset
    if args.number_of_rows:
        NUMBER_OF_ROWS_TO_PROCESS = args.number_of_rows
    if args.retry_attempts:
        MAX_NUMBER_OF_TRY_PER_PROMPT = args.retry_attempts
    if args.prepare_prompts:
        PREPARE_PROMPTS = args.prepare_prompts
    if args.system_prompt:
        SYSTEM_PROMPT = args.system_prompt
    if args.system_prompt_path:
        with open(args.system_prompt_path, 'r') as f:
            SYSTEM_PROMPT = f.read()
    if args.output_path:
        OUTPUT_PATH = args.output_path
    if args.queries_path:
        QUERIES_PATH = args.queries_path
    if args.prediction_size:
        NUMBER_OF_TOKEN_TO_PREDICT = args.prediction_size
    if args.temperature:
        TEMPERATURE = args.temperature

def print_header(args):
    if args.verbose:
        print(LINE)
        print(f"Reverse Prompt generation v{VERSION}".center(LINE_WIDTH, ' '))
        print(LINE)
        
        print("Script Parameters")
        print("  PROVIDER".ljust(30), PROVIDER)
        if PROVIDER == "SERVER":
            print("  SERVER_ADDR".ljust(30), SERVER_ADDR)
            print("  SERVER_PORT".ljust(30), SERVER_PORT)
            print("  SERVER_COMPLETION_ENDPOINT".ljust(30), SERVER_COMPLETION_ENDPOINT)
        elif PROVIDER == "CTRANSFORMERS":
            print("  MODEL_PATH".ljust(30), CT_MODEL_PATH)
        print("  STARTING_ROW_OFFSET".ljust(30), STARTING_ROW_OFFSET)
        print("  NUMBER_OF_ROWS_TO_PROCESS".ljust(30), NUMBER_OF_ROWS_TO_PROCESS)
        print("  SYSTEM_PROMPT".ljust(30), SYSTEM_PROMPT)
        print("  MAX_NUMBER_OF_TRY_PER_PROMPT".ljust(30), MAX_NUMBER_OF_TRY_PER_PROMPT)
        print("  RETRY_IF_ANSWER_CONTAINS".ljust(30), "\"" + "\", \"".join(RETRY_IF_ANSWER_CONTAINS) + "\"")
        print("  PREPARE_PROMPTS".ljust(30), PREPARE_PROMPTS)
        print("  OUTPUT_PATH".ljust(30), OUTPUT_PATH)
        print("  CLEANED_QUERIES_PATH".ljust(30), QUERIES_PATH)
        print("  NUMBER_OF_TOKEN_TO_PREDICT".ljust(30), NUMBER_OF_TOKEN_TO_PREDICT)
        print("  TEMPERATURE".ljust(30), TEMPERATURE)

def print_additional_infos(args, df_cleaned_queries):
    if args.verbose:
        print("Additional Information")
        print("  Number of rows to process".ljust(30), NUMBER_OF_ROWS_TO_PROCESS if NUMBER_OF_ROWS_TO_PROCESS > 0 else len(df_cleaned_queries) - STARTING_ROW_OFFSET)
        print(LINE)

if __name__ == '__main__':
    args = parse_script_arguments()
    update_script_parameters(args)
    
    provider = None
    if PROVIDER == "SERVER":
        provider = ServerProvider()
    elif PROVIDER == "CTRANSFORMERS":
        provider = CTransformersProvider()
    
    print_header(args)
        
    df_cleaned_queries = load_and_prepare_queries(basic_prompt)

    print_additional_infos(args, df_cleaned_queries)
    
    if args.verbose:
        print("Starting execution.")
        print("Compiling regex... ", end="")
    compiled_regex_prompts_extractor = re.compile(r'\"[A-Z].*\"', flags=0)
    if args.verbose:
        print("Done.")
    
    prompts = df_cleaned_queries["prompt"]
    num_tokens = df_cleaned_queries['num_tokens']
    skipped_rows = []

    if args.verbose:
        print("Generating reverse prompts... ")
    last_row_number = len(df_cleaned_queries) if NUMBER_OF_ROWS_TO_PROCESS <= 0 else STARTING_ROW_OFFSET + NUMBER_OF_ROWS_TO_PROCESS
    for row_index in tqdm(range(STARTING_ROW_OFFSET, last_row_number)):
        number_of_try_left = MAX_NUMBER_OF_TRY_PER_PROMPT
        found_results = False
        too_much_tokens = False
        
        prompt = prompts.iat[row_index]
        num_token = num_tokens.iat[row_index]
        
        if num_token > CT_CONTEXT_LENGTH:
            too_much_tokens = True
        else:
            data_json = prepare_request_payload(prompt)
        
        while number_of_try_left != 0 and not (found_results or not too_much_tokens):    
            provider.query(data_json)
            
            if args.print_answers:
                print(provider.get_answer())
            
            result = compiled_regex_prompts_extractor.findall(provider.get_answer())
            
            if args.print_results:
                print(result)
            
            if not are_results_acceptable(result):
                number_of_try_left -= 1
                continue
            
            found_results = True
            df_cleaned_queries['result'].iat[row_index] = result
            df_cleaned_queries['full_answer'].iat[row_index] = provider.get_full_answer()
        
        if not found_results and not args.quiet:
            print(f"No results found for: {df_cleaned_queries.iloc[row_index].name}")
            skipped_rows.append(df_cleaned_queries.iloc[row_index].name)
        elif too_much_tokens and not args.quiet:
            print(f"Prompt has too much token at row: {df_cleaned_queries.iloc[row_index].name}")
            skipped_rows.append(df_cleaned_queries.iloc[row_index].name)
    
    if args.verbose:
        print("Printing dataset... ", end="")
    dataframe_json_dump = df_cleaned_queries.iloc[STARTING_ROW_OFFSET:last_row_number].to_json()
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    with open(f"{OUTPUT_PATH}{datetime.datetime.now().strftime('%Y%m%d-%H%M')}_{STARTING_ROW_OFFSET:04}-{last_row_number:04}_generated_prompts.json", "w") as outfile:
        outfile.write(dataframe_json_dump)
    
    if args.verbose:
        print("Done.")

    if len(skipped_rows) > 0:
        if args.verbose:
            print("Printing skipped rows... ", end="")
            
        skipped_rows_json_dump = json.dumps(skipped_rows)
        with open(f"{OUTPUT_PATH}{datetime.datetime.now().strftime('%Y%m%d-%H%M')}_skipped_rows.json", "w") as outfile:
            outfile.write(skipped_rows_json_dump)
            
        if args.verbose:
            print("Done.")
            
    if args.verbose:
        print("Printing execution summary file... ", end="")
    
    summary = {
        "PROVIDER": PROVIDER,
        "SERVER_ADDR": SERVER_ADDR,
        "SERVER_PORT": SERVER_PORT,
        "SERVER_COMPLETION_ENDPOINT": SERVER_COMPLETION_ENDPOINT,
        "MODEL_PATH": CT_MODEL_PATH,
        "STARTING_ROW_OFFSET": STARTING_ROW_OFFSET,
        "NUMBER_OF_ROWS_TO_PROCESS": NUMBER_OF_ROWS_TO_PROCESS,
        "MAX_NUMBER_OF_TRY_PER_PROMPT": MAX_NUMBER_OF_TRY_PER_PROMPT,
        "RETRY_IF_ANSWER_CONTAINS": RETRY_IF_ANSWER_CONTAINS,
        "SYSTEM_PROMPT": SYSTEM_PROMPT,
        "OUTPUT_PATH": OUTPUT_PATH,
        "CLEANED_QUERIES_PATH": QUERIES_PATH,
        "NUMBER_OF_TOKEN_TO_PREDICT": NUMBER_OF_TOKEN_TO_PREDICT,
        "TEMPERATURE": TEMPERATURE,
        "NUMBER_OF_SKIPPED_ROWS": len(skipped_rows)
    }
    
    summary_json_dump = json.dumps(summary)
    with open(f"{OUTPUT_PATH}{datetime.datetime.now().strftime('%Y%m%d-%H%M')}_execution_summary.json", "w") as outfile:
            outfile.write(summary_json_dump)
    
    if args.verbose:
        print("Done.")
        
    if not args.quiet:
        print("Execution successfully ended.")