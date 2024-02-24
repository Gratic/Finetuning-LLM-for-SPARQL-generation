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

def generate_prompt(config: configparser.ConfigParser):
    return_code = subprocess.run(["python3", "-m", "libsparqltotext",
                                  ""])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Dataset Pipeline",
                                 description="From the raw dataset to the finetunning datasets.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the (ini) config file.")
    parser.add_argument("-i", "--id", type=str, required=True, help="ID of the dataset pipeline.")
    parser.add_argument("-o", "--output", type=str, help="Where the script should save results.", default="./outputs/dataset_pipeline/")
    parser.add_argument("-d", "--debug", action="store_true", help="Put the script in debug mode.")

    args = parser.parse_args()
    
    config = read_config_file(args)
    create_folder_structure(args)
    
    if args.debug:
        print("Config sections are:")    
        print(config.sections())
    
    dataset_path = Path(config['Dataset'].get("dataset_path"))
    # TODO: uncomment
    # check_initial_data(dataset_path)
    
    script_path = check_and_get_script_path(config)
    
    
    # TODO: generate prompts for them using libsparqltotext
    # TODO: execute the dataset using execute_queries.py
    # TODO: Merge them both together?
    # TODO: Only keep rows that has a succesful execution
    # TODO: Split the dataset in 3 using generate_finetune_dataset.py