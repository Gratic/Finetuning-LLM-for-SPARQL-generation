import sys
from pathlib import Path
sys.path.append(Path("modules").absolute().__str__())

from pathlib import Path
from typing import List
import argparse
import configparser
import pandas as pd
import subprocess
import time
from data_utils import load_dataset

def read_config_file(args):
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"There is no config file at path: {str(config_path)}")
    
    config = configparser.ConfigParser(allow_no_value=True, converters={"list": lambda x: [i.strip() for i in x.split(',')]})
    config.read(args.config)
    return config

def create_folder_structure(args):
    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)
    id_folder = output_folder / args.id
    id_folder.mkdir(parents=True, exist_ok=args.debug or args.continue_execution)
    
    return (output_folder, id_folder)

def check_columns(dataset: pd.DataFrame, columns: List[str]):
    if not all([col in dataset.columns for col in columns]):
        raise ValueError("Not all the required columns were present in the dataset.")
    
def check_initial_data(dataset_path: Path):
    dataset = load_dataset(dataset_path)
    
    check_columns(dataset, [
        "query",
        "context",
        "description"
    ])
    
def check_and_get_script_path(config: configparser.ConfigParser):
    scripts_paths = {}
    
    for key in config['Scripts'].keys():
        path = Path(config['Scripts'].get(key))
        
        if not path.exists():
            raise FileNotFoundError(f"The {key} was not found at this path: {str(path)}")
        
        scripts_paths.update({key: path})
        
    return scripts_paths

def templatize_queries(id_folder: Path, templatize_script: Path, config: configparser.ConfigParser, dataset_path: Path):
    dataset_templated = id_folder / f"{id_folder.name}-templated.json"

    if dataset_templated.exists():
        return dataset_templated
    
    return_code = subprocess.run(["python3", templatize_script,
                                  "--input", str(dataset_path),
                                  "--column", "query",
                                  "--out-column", "query_templated",
                                  "--output", str(id_folder),
                                  "--save-name", dataset_templated.stem,
                                  "--prefix"])
    
    if return_code.returncode != 0:
        print(f"Failed to templatize prompts.")
        exit()
        
    if not dataset_templated.exists():
        raise FileNotFoundError(f"The resulting dataset with prompts was not found: {str(dataset_templated)}")
    
    return dataset_templated

def generate_prompts(id_folder: Path, config: configparser.ConfigParser, dataset_path: Path, prefix: str, query_column:str):
    launch_server = config["Provider LLAMACPP"].getboolean("launch_server")
    if launch_server:
        llama_server = launch_llama_server(config)
    
    provider_config = config["Prompt Generation.Provider Configuration"]
    dataset_config = config["Prompt Generation.Dataset Configuration"]
    generation_config = config["Prompt Generation.Generation Configuration"]
    
    checkpoint_folder = id_folder / f"{prefix}generation_checkpoint"
    checkpoint_folder.mkdir(parents=True, exist_ok=True)
    
    dataset_with_prompts = id_folder / f"{prefix}{id_folder.name}-generated_prompt_{query_column}.json"
    if dataset_with_prompts.exists():
        return dataset_with_prompts
    
    return_code = subprocess.run(["python3", "-m", "modules.libsparqltotext",
                                  "--queries-path", str(dataset_path),
                                  "--provider", provider_config.get("provider"),
                                  "--server-address", provider_config.get("server_address"),
                                  "--server-port", provider_config.get("server_port"),
                                  "--completion-endpoint", provider_config.get("completion_endpoint"),
                                  "--tokenizer-endpoint", provider_config.get("tokenizer_endpoint"),
                                  "--model-path", provider_config.get("model_path"),
                                  "--context-length", provider_config.get("context_length"),
                                  "--generation", dataset_config.get("generation"),
                                  "--offset", dataset_config.get("offset"),
                                  "--number-of-rows", dataset_config.get("number_of_rows"),
                                  "--target-rows", dataset_config.get("target_rows"),
                                  "--retry-attempts", dataset_config.get("retry_attempts"),
                                  "--prepare-prompts", generation_config.get("prepare_prompts"),
                                  "--template", generation_config.get("prompt_template"),
                                  "--system-prompt", generation_config.get("system_prompt"),
                                  "--prompt", generation_config.get("prompt"),
                                  "--leading-answer-prompt", generation_config.get("leading_answer_prompt"),
                                  "--prediction-size", generation_config.get("prediction_size"),
                                  "--temperature", generation_config.get("temperature"),
                                  "--save-identifier", str(id_folder.name),
                                  "--checkpoint-path", str(checkpoint_folder),
                                  "--output-path", str(id_folder),
                                  "--save-name", dataset_with_prompts.stem,
                                  "--prefix", prefix,
                                  "--query-column", query_column
                                  ])

    if launch_server:
        terminate_process(llama_server)

    if return_code.returncode != 0:
        print(f"Failed to generate prompt.")
        exit()
        
    if not dataset_with_prompts.exists():
        raise FileNotFoundError(f"The resulting dataset with prompts was not found: {str(dataset_with_prompts)}")
    
    return dataset_with_prompts

def execute_queries(id_folder: Path, execution_script: Path, dataset_path: Path, config: configparser.ConfigParser):
    execution_config = config["Query Execution"]
    
    dataset_with_prompts_executed = id_folder / f"{id_folder.name}-generated_prompt-executed.parquet.gzip"
    if dataset_with_prompts_executed.exists():
        return dataset_with_prompts_executed
    
    return_code = subprocess.run(["python3", execution_script,
                                  "--dataset", str(dataset_path),
                                  "--column-name", "query",
                                  "--timeout", execution_config.get("timeout"),
                                  "--limit", execution_config.get("per_query_answer_limit"), # If limit == 0, no LIMIT clause will be automatically added, however if present already will not be removed.
                                  "--output", str(id_folder),
                                  "--save-name", dataset_with_prompts_executed.stem])
    
    if return_code.returncode != 0:
        print(f"Failed to execute queries.")
        exit()
    
    if not dataset_with_prompts_executed.exists():
        raise FileNotFoundError(f"The resulting dataset with executed queries was not found: {str(dataset_with_prompts_executed)}")
    
    return dataset_with_prompts_executed

def split_dataset(id_folder: Path, split_dataset_script: Path, dataset_path: Path, config: configparser.ConfigParser):
    split_name = f"{id_folder.name}-split"
    return_code = subprocess.run(["python3", str(split_dataset_script),
                                  "--input", str(dataset_path),
                                  "--keep-working",
                                  "--output", str(id_folder),
                                  "--save-name", split_name,
                                  "--random-seed", str(random_seed),
                                  ])

    if return_code.returncode != 0:
        print(f"Failed to split dataset.")
        exit()
        
    split_train = id_folder / f"{split_name}_train.pkl"
    split_valid = id_folder / f"{split_name}_valid.pkl"
    split_test = id_folder / f"{split_name}_test.pkl"
    gold_train = id_folder / f"gold_{split_name}_train.json"
    gold_valid = id_folder / f"gold_{split_name}_valid.json"
    gold_test = id_folder / f"gold_{split_name}_test.json"
    
    if not all([path.exists() for path in [split_train, split_valid, split_test, gold_train, gold_valid, gold_test]]):
        raise FileNotFoundError("The splits file are not found.")
    
    return split_train, split_valid, split_test

def launch_llama_server(config: configparser.ConfigParser):
    server_config = config["Provider LLAMACPP"]
    server_path = Path(server_config.get("server_path"))
    model_path = Path(server_config.get("model_path"))
    
    if not server_path.exists():
        FileNotFoundError(f"The LLAMA.CPP server executable was not found at: {str(server_path)}")
    
    if not model_path.exists():
        FileNotFoundError(f"The model was not found at: {str(model_path)}")
    
    process = subprocess.Popen([server_path.absolute(),
                      "-m", model_path.absolute(),
                      "-ngl", server_config.get("n_layer_on_gpu"),
                      "-c", server_config.get("context_length")])
    
    time.sleep(server_config.getint("delay"))
    
    return process
    
def terminate_process(process: subprocess.Popen):
    process.terminate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Dataset Pipeline",
                                 description="From the raw dataset to the finetunning datasets.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the (ini) config file.")
    parser.add_argument("-i", "--id", type=str, required=True, help="ID of the dataset pipeline.")
    parser.add_argument("-o", "--output", type=str, help="Where the script should save results.", default="./outputs/dataset_pipeline/")
    parser.add_argument("-d", "--debug", action="store_true", help="Put the script in debug mode.")
    parser.add_argument("-ce", "--continue-execution", action="store_true", help="Take back from the execution of the script.")

    args = parser.parse_args()
    
    config = read_config_file(args)
    
    random_seed = config['Execution'].getint("random_seed")
    
    output_folder, id_folder = create_folder_structure(args)
    
    if args.debug:
        print("Config sections are:")    
        print(config.sections())
    
    dataset_path = Path(config['Dataset'].get("dataset_path"))
    check_initial_data(dataset_path)
    
    script_path = check_and_get_script_path(config)
    
    dataset_templated = templatize_queries(
        id_folder=id_folder,
        templatize_script=script_path['templatize_script'],
        config=config,
        dataset_path=dataset_path
    )
    
    print(f"Dataset with templated prompt can be found at: '{str(dataset_templated)}'.")
    
    dataset_with_basic_prompts = generate_prompts(
        id_folder=id_folder,
        config=config,
        dataset_path=dataset_templated,
        prefix=config['Prompt Generation.Using Basic'].get("prefix"),
        query_column=config['Prompt Generation.Using Basic'].get("query_column"),
        )

    print(f"Dataset with basic prompt can be found at: '{str(dataset_with_basic_prompts)}'.")
    
    dataset_with_templated_prompts = generate_prompts(
        id_folder=id_folder,
        config=config,
        dataset_path=dataset_with_basic_prompts,
        prefix=config['Prompt Generation.Using Templated'].get("prefix"),
        query_column=config['Prompt Generation.Using Templated'].get("query_column"),
        )

    print(f"Dataset with templated prompt can be found at: '{str(dataset_with_templated_prompts)}'.")
    
    dataset_with_prompts_executed = execute_queries(
        id_folder=id_folder,
        execution_script=script_path['query_execution_script'],
        dataset_path=dataset_with_templated_prompts,
        config=config
        )
    
    print(f"Dataset with prompt executed can be found at: '{str(dataset_with_prompts_executed)}'.")
    
    split_train, split_valid, split_test = split_dataset(
        id_folder=id_folder, 
        split_dataset_script=script_path["split_dataset_script"], 
        dataset_path=dataset_with_prompts_executed, 
        config=config
        )
    
    print(f"Train split can be found at: '{str(split_train)}'.")
    print(f"Train split can be found at: '{str(split_valid)}'.")
    print(f"Train split can be found at: '{str(split_test)}'.")
    
    print("Execution succesful.")