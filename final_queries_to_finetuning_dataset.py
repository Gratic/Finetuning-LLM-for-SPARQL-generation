import pandas as pd
import argparse
import subprocess
import configparser
from pathlib import Path
from typing import List

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
    id_folder.mkdir(parents=True, exist_ok=args.debug)
    
    return (output_folder, id_folder)

def read_pandas_dataset(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"The dataset path is not correct: {str(path)}")
    
    return pd.read_json(path)

def check_columns(dataset: pd.DataFrame, columns: List[str]):
    if not all([col in dataset.columns for col in columns]):
        raise ValueError("Not all the required columns were present in the dataset.")
    
def check_initial_data(dataset_path: Path):
    dataset = read_pandas_dataset(dataset_path)
    
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

def generate_prompts(id_folder: Path, config: configparser.ConfigParser, dataset_path: Path):
    
    provider_config = config["Prompt Generation.Provider Configuration"]
    dataset_config = config["Prompt Generation.Dataset Configuration"]
    generation_config = config["Prompt Generation.Generation Configuration"]
    
    checkpoint_folder = id_folder / "generation_checkpoint"
    checkpoint_folder.mkdir(parents=True, exist_ok=True)
    
    dataset_with_prompts = f"{id_folder.name}-generated_prompt"
    return_code = subprocess.run(["python3", "-m", "libsparqltotext",
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
                                  "--save-name", dataset_with_prompts])

    if return_code.returncode != 0:
        print(f"Failed to generate prompt.")
        exit()
        
    dataset_with_prompts = id_folder / f"{dataset_with_prompts}.json"
    if not dataset_with_prompts.exists():
        raise FileNotFoundError(f"The resulting dataset with prompts was not found: {str(dataset_with_prompts)}")
    
    return dataset_with_prompts

def execute_queries(id_folder: Path, execution_script: Path, dataset_path: Path, config: configparser.ConfigParser):
    execution_config = config["Query Execution"]
    
    dataset_with_prompts_executed = f"{id_folder.name}-generated_prompt-executed"
    return_code = subprocess.run(["python3", execution_script,
                                  "--dataset", str(dataset_path),
                                  "--column-name", "query",
                                  "--timeout", execution_config.get("timeout"),
                                  "--limit", execution_config.get("per_query_answer_limit"),
                                  "--output", str(id_folder),
                                  "--save-name", dataset_with_prompts_executed])
    
    if return_code.returncode != 0:
        print(f"Failed to execute queries.")
        exit()
    
    dataset_with_prompts_executed = id_folder / f"{dataset_with_prompts_executed}.parquet.gzip"
    if not dataset_with_prompts_executed.exists():
        raise FileNotFoundError(f"The resulting dataset with executed queries was not found: {str(dataset_with_prompts_executed)}")
    
    return dataset_with_prompts_executed

def split_dataset(id_folder: Path, split_dataset_script: Path, dataset_path: Path, config: configparser.ConfigParser):
    split_name = f"{id_folder.name}-split"
    return_code = subprocess.run(["python3", str(split_dataset_script),
                                  "--input", str(dataset_path),
                                  "--output", str(id_folder),
                                  "--save-name", split_name])

    if return_code.returncode != 0:
        print(f"Failed to split dataset.")
        exit()
        
    split_train = id_folder / f"{split_name}_train.pkl"
    split_valid = id_folder / f"{split_name}_valid.pkl"
    split_test = id_folder / f"{split_name}_test.pkl"
    
    if not all([path.exists() for path in [split_train, split_valid, split_test]]):
        raise FileNotFoundError("The splits file are not found.")
    
    return split_train, split_valid, split_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Dataset Pipeline",
                                 description="From the raw dataset to the finetunning datasets.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the (ini) config file.")
    parser.add_argument("-i", "--id", type=str, required=True, help="ID of the dataset pipeline.")
    parser.add_argument("-o", "--output", type=str, help="Where the script should save results.", default="./outputs/dataset_pipeline/")
    parser.add_argument("-d", "--debug", action="store_true", help="Put the script in debug mode.")

    args = parser.parse_args()
    
    config = read_config_file(args)
    output_folder, id_folder = create_folder_structure(args)
    
    if args.debug:
        print("Config sections are:")    
        print(config.sections())
    
    dataset_path = Path(config['Dataset'].get("dataset_path"))
    check_initial_data(dataset_path)
    
    script_path = check_and_get_script_path(id_folder, config, dataset_path)
    
    dataset_with_prompts = generate_prompts(
        id_folder=id_folder,
        config=config,
        dataset_path=dataset_path
        )

    print(f"Dataset with prompt can be found at: '{str(dataset_with_prompts)}'.")
    
    dataset_with_prompts_executed = execute_queries(
        id_folder=id_folder,
        execution_script=script_path['query_execution_script'],
        dataset_path=dataset_with_prompts,
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