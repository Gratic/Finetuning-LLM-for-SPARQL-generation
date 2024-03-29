import argparse
import itertools
import json
import logging
import os
import subprocess
from typing import Dict
import configparser

def file_exists_or_raise(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The script was not found in: {file_path}")

def generate_name_from_dict(params_dict: Dict, abbrev_dict: Dict):
    return "-".join([abbrev_dict[key] + keep_only_alphanum_chars(str(params_dict[key])) for key in filter(lambda x: abbrev_dict[x] != "no", list(params_dict.keys()))])

def generate_folder_structure(args):
    os.makedirs(args.output, exist_ok=True)
    
    batch_run_folder = os.path.join(args.output, args.id)
    generation_folder = os.path.join(batch_run_folder, "generation")
    execution_folder = os.path.join(batch_run_folder, "execution")
    evaluation_folder = os.path.join(batch_run_folder, "evaluation")
    
    if os.path.exists(batch_run_folder) and not args.recover:
        raise Exception(f"A previous batch run has been executed with this id: {args.id} .")
    
    os.makedirs(batch_run_folder, exist_ok=args.recover)
    os.makedirs(generation_folder, exist_ok=args.recover)
    os.makedirs(execution_folder, exist_ok=args.recover)
    os.makedirs(evaluation_folder, exist_ok=args.recover)
    return batch_run_folder,generation_folder,execution_folder,evaluation_folder

def setup_logging(args, batch_run_folder):
    numeric_log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_log_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}.")
    
    log_file = os.path.join(batch_run_folder, "outputs.log")
    logging.basicConfig(filename=log_file, level=numeric_log_level)
    return log_file

def keep_only_alpha_chars(string: str):
    return "".join(filter(lambda x: x.isalpha(), string))

def keep_only_alphanum_chars(string: str):
    return "".join(filter(lambda x: x.isalnum(), string))

if __name__ == "__main__":
    """
    This script is an "orchestrator". It takes a config file, do some preprocessing (gold/test dataset execution) and then the training of multiple LLMs from it.
    If the training is succesful (sft_peft.py returns 0) it will:
    - generate queries using it,
    - execute the generated queries on wikidata's SPARQL endpoint,
    - evaluate it.
    
    When all evaluations are dones, the script launches a last script that takes all the evaluations and concatenates it in one file.
    """
    
    parser = argparse.ArgumentParser(prog="Batch run LLM training to LLM evaluation on SPARQL Dataset",
                                     description="Orchestrate the run of multiple script to train an LLM and evaluate it.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the (ini) config file.")
    parser.add_argument("-i", "--id", type=str, required=True, help="ID of the batch run.")
    parser.add_argument("-o", "--output", type=str, help="Where the batch run should save results.", default="./outputs/batch_run/")
    parser.add_argument("-log", "--log-level", type=str, help="Logging level (debug, info, warning, error, critical).", default="warning")
    parser.add_argument("-r", "--recover", action="store_true", help="Try to recover a failed run from the id.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"The mandatory config file was not found at: {args.config} .")
    
    preprocessing_gold_script_path = "scripts/preprocess_gold_dataset_for_evaluation.py"
    training_script_path = "scripts/sft_peft.py"
    merging_script_path = "scripts/merge_adapters.py"
    libwikidatallm_path = "modules.libwikidatallm"
    executing_queries_script_path = "scripts/execute_queries.py"
    evaluation_script_path = "scripts/evaluation_bench.py"
    concatenation_script_path = "scripts/concatenate_evaluations.py"
    
    map(file_exists_or_raise, [
        preprocessing_gold_script_path, 
        training_script_path,
        merging_script_path,
        libwikidatallm_path,
        executing_queries_script_path,
        evaluation_script_path,
        concatenation_script_path
        ])
    
    batch_run_folder, generation_folder, execution_folder, evaluation_folder = generate_folder_structure(args)
    
    log_file = setup_logging(args, batch_run_folder)
    
    logging.info("Loading config dataset.")
    config = configparser.ConfigParser(allow_no_value=True, converters={"list": lambda x: [i.strip() for i in x.split(',')]})
    config.read(args.config)
    
    random_seed = config['Execution'].getint('random_seed')
    
    # 0.1) Execute the test dataset against wikidata API
    if not os.path.exists(config["Datasets"]["test"]):
        raise FileNotFoundError(f"The test dataset wasn't found at: {config['Datasets']['test']}")
    if not os.path.exists(config["Datasets"]["train"]):
        raise FileNotFoundError(f"The train dataset wasn't found at: {config['Datasets']['train']}")
    if not os.path.exists(config["Datasets"]["valid"]):
        raise FileNotFoundError(f"The valid dataset wasn't found at: {config['Datasets']['valid']}")
    
    preprocessed_gold_dataset = None
    if config['Execution'].getboolean('do_preprocess_gold'):
        logging.info("Executing the test dataset.")
        gold_execute_name = f"gold_executed"
        gold_execute_queries_return = subprocess.run(["python3", executing_queries_script_path,
                                                    "--dataset", config['Datasets']["test"],
                                                    "--column-name", "target_raw",
                                                    "--timeout", str(60),
                                                    "--limit", str(10),
                                                    "--output", execution_folder,
                                                    "--save-name", gold_execute_name,
                                                    ])
        
        if gold_execute_queries_return.returncode != 0:
            logging.error(f"Failed to execute gold queries: {gold_execute_name}.")
            print(f"Failed to execute gold queries: {gold_execute_name}.")
            exit()
            
        gold_executed_queries_path = os.path.join(execution_folder, f"{gold_execute_name}.parquet.gzip")
        
        # 0.2) To reduce the number of computations during evaluation, preprocess the test dataset
        logging.info("Preprocessing the gold dataset.")
        gold_evaluated = "preprocessed_gold"
        preprocess_gold_return = subprocess.run(["python3", preprocessing_gold_script_path,
                                        "--gold", gold_executed_queries_path,
                                        "--output", batch_run_folder,
                                        "--save-name", gold_evaluated,
                                        "--log-level", args.log_level,
                                        "--log-file", log_file
                                        ])
        
        if preprocess_gold_return.returncode != 0:
            logging.error(f"Failed to preprocess gold.")
            print(f"Failed to preprocess gold.")
            exit()
        
        preprocessed_gold_dataset = os.path.join(batch_run_folder, f"{gold_evaluated}.json")
    else:
        if not os.path.exists(config["Execution"]["preprocess_gold_path"]):
            raise FileNotFoundError(f"The preprocessed gold dataset file was not found at given path: {config['Execution']['preprocess_gold_path']}")
        logging.info("Loading already executed gold dataset.")
        preprocessed_gold_dataset = config["Execution"]["preprocess_gold_path"]
    
    training_hyperparameters = [
        json.loads(config['Models'].get('models')),
        config['Training Hyperparameters'].getlist("lora_r_value"),
        config['Training Hyperparameters'].getlist("lora_dropout"),
        config['Training Hyperparameters'].getlist("batch_size"),
        config['Training Hyperparameters'].getlist("packing"),
        config['Training Hyperparameters'].getlist("neft_tune_alpha"),
        config['Training Hyperparameters'].getlist("pipeline_type"),
        config['Training Hyperparameters'].getlist("input_type"),
    ]
    num_epochs = config['Training Hyperparameters'].getint("epochs")
    possible_target_columns = config["Pipeline Types To Target Columns"]
    possible_input_columns = config["Input Types to Input Columns"]
    
    config['Evaluation Hyperparameters']['start_tag'] = config['Evaluation Hyperparameters'].get('start_tag').replace('\\n', '\n')
    config['Evaluation Hyperparameters']['end_tag'] = config['Evaluation Hyperparameters'].get('end_tag').replace('\\n', '\n')
    
    logging.info("Starting the training and evaluation loop.")
    for model_obj, rvalue, lora_dropout, batch_size, packing, neft_tune_alpha, pipeline_type, input_type in itertools.product(*training_hyperparameters):
        # 1) Train an LLM (sft_peft.py)
        packing = int(packing)

        train_params_dict = {
            "lora_r_value": rvalue,
            "lora_dropout": lora_dropout,
            "batch_size": batch_size,
            "packing": packing,
            "neft_tune_alpha": neft_tune_alpha,
            "num_epochs": num_epochs
        }
        
        modified_start_tag = keep_only_alpha_chars(config['Evaluation Hyperparameters']['start_tag'])
        full_model_name = f"{model_obj['name']}_{generate_name_from_dict(train_params_dict, config['Training Hyperparameters Name Abbreviations'])}-{pipeline_type}-{input_type}-st{modified_start_tag}"
        
        adapters_model_path = os.path.join(args.output, f"{full_model_name}_adapters")
        
        if not os.path.exists(adapters_model_path):
            logging.info(f"Starting LLM Training: {model_obj['name']=}, {rvalue=}, {lora_dropout=}, {batch_size=}, {bool(packing)=}, {neft_tune_alpha=}")
            print(f"Starting LLM Training: {model_obj['name']=}, {rvalue=}, {lora_dropout=}, {batch_size=}, {bool(packing)=}, {neft_tune_alpha=}")
            use_accelerate = config["Execution"].getboolean("use_accelerate")
            print(f"Using accelerate: {str(use_accelerate)}")
            training_return = subprocess.run((["accelerate", "launch"] if use_accelerate else ["python3"]) + [training_script_path,
                                            "--model", model_obj['path'],
                                            "--train-data", config["Datasets"]["train"],
                                            "--target-column", possible_target_columns[pipeline_type],
                                            "--input-column", possible_input_columns[input_type],
                                            "--valid-data", config["Datasets"]["valid"],
                                            "--start-tag", str(config['Evaluation Hyperparameters']['start_tag']),
                                            "--end-tag", str(config['Evaluation Hyperparameters']['end_tag']),
                                            "--rvalue", str(rvalue),
                                            "--lora-dropout", str(lora_dropout),
                                            "--batch-size", str(batch_size),
                                            "--gradient-accumulation", str(4),
                                            "--packing", str(packing),
                                            "--neft-tune-alpha", str(neft_tune_alpha),
                                            "--epochs", str(num_epochs),
                                            "--output", args.output,
                                            "--save-name", full_model_name,
                                            "--run-name", f"{args.id}-{full_model_name}",
                                            "--save-adapters",
                                            "--log-level", args.log_level,
                                            "--log-file", log_file,
                                            "--random-seed", str(random_seed)
                                            ])
            
            if training_return.returncode != 0:
                logging.error(f"Failed to train: {full_model_name}.")
                print(f"Failed to train: {full_model_name}.")
                continue
        else:
            logging.info(f"Recovered adapter for: {model_obj['name']=}, {rvalue=}, {lora_dropout=}, {batch_size=}, {bool(packing)=}, {neft_tune_alpha=}")
            print(f"Recovered adapter for: {model_obj['name']=}, {rvalue=}, {lora_dropout=}, {batch_size=}, {bool(packing)=}, {neft_tune_alpha=}")
        
        # 2) Generate sparql queries using libwikidatallm
        generation_name = f"{full_model_name}_{generate_name_from_dict(config['Evaluation Hyperparameters'], config['Evaluation Hyperparameters Name Abbreviations'])}"
        generated_queries_path = os.path.join(generation_folder, f"{generation_name}.parquet.gzip")
        
        if not os.path.exists(generated_queries_path):
            logging.info(f"Generating SPARQL queries: model={full_model_name}, temperature={config['Evaluation Hyperparameters']['temperature']}, top-p={config['Evaluation Hyperparameters']['top_p']}")        
            print(f"Generating SPARQL queries: model={full_model_name}, temperature={config['Evaluation Hyperparameters']['temperature']}, top-p={config['Evaluation Hyperparameters']['top_p']}")
            generate_queries_return = subprocess.run(["python3", "-m", libwikidatallm_path,
                                                    "--data", config["Datasets"]["test"],
                                                    # We could also try with the other column here but is it pertinent?
                                                    "--column-name", possible_input_columns[input_type],
                                                    "--model", model_obj['path'],
                                                    "--adapters", adapters_model_path,
                                                    "--tokenizer", model_obj['path'],
                                                    "--context-length", str(model_obj['context_length']),
                                                    "--engine", config["Evaluation Hyperparameters"]["engine"],
                                                    "--pipeline", pipeline_type,
                                                    "--start-tag", str(config['Evaluation Hyperparameters']['start_tag']),
                                                    "--end-tag", str(config['Evaluation Hyperparameters']['end_tag']),
                                                    "--decoding", str(config["Evaluation Hyperparameters"]["decoding"]),
                                                    "--temperature", str(config['Evaluation Hyperparameters']['temperature']),
                                                    "--topp", str(config['Evaluation Hyperparameters']['top_p']),
                                                    "--num-tokens", str(config['Evaluation Hyperparameters']['num_tokens']),
                                                    "--output", generation_folder,
                                                    "--save-name", generation_name,
                                                    "--tqdm",
                                                    "--random-seed", str(random_seed),
                                                    ])
            
            if generate_queries_return.returncode != 0:
                logging.error(f"Failed to generate queries: {generation_name}.")
                print(f"Failed to generate queries: {generation_name}.")
                continue
        
            if not os.path.exists(generated_queries_path):
                raise FileNotFoundError(f"The generated queries were not found: {generated_queries_path}.")
        else:
            logging.info(f"Recovered generated queries for: {generation_name}")
            print(f"Recovered generated queries for: {generation_name}")
            
        # 3) Execute queries on wikidata (execute_queries.py)        
        execute_name = f"{generation_name}_executed"
        executed_queries_path = os.path.join(execution_folder, f"{execute_name}.parquet.gzip")
        
        if not os.path.exists(executed_queries_path):
            logging.info("Executing generated queries on Wikidata's SPARQL Endpoint.")
            print("Executing generated queries on Wikidata's SPARQL Endpoint.")
            execute_queries_return = subprocess.run(["python3", executing_queries_script_path,
                                                    "--dataset", generated_queries_path,
                                                    "--column-name", "output",
                                                    "--timeout", str(60),
                                                    "--limit", str(10),
                                                    "--output", execution_folder,
                                                    "--save-name", execute_name,
                                                    ])
            
            if execute_queries_return.returncode != 0:
                logging.error(f"Failed to execute queries: {execute_name}.")
                print(f"Failed to execute queries: {execute_name}.")
                continue
            
            
            if not os.path.exists(executed_queries_path):
                raise FileNotFoundError(f"The executed queries were not found: {executed_queries_path}.")
        else:
            logging.info(f"Recovered execution: {execute_name}")
            print(f"Recovered execution: {execute_name}")
            
        # 4) Evaluate the LLM
        logging.info(f"Evaluating {full_model_name}.")
        print(f"Evaluating {full_model_name}.")
        
        evaluate_name = f"{execute_name}_evaluated"
        evaluate_return = subprocess.run(["python3", evaluation_script_path,
                                          "--dataset", executed_queries_path,
                                          "--preprocess-gold", preprocessed_gold_dataset,
                                          "--model", full_model_name,
                                          "--output", evaluation_folder,
                                          "--save-name", evaluate_name,
                                          "--log-level", args.log_level,
                                          "--log-file", log_file
                                          ])

        if evaluate_return.returncode != 0:
            logging.error(f"Failed to evaluate llm: {evaluate_name}.")
            print(f"Failed to evaluate llm: {evaluate_name}.")
            continue   
    
    logging.info("Concatenating all evaluations.")
    print("Concatenating all evaluations.")
    
    subprocess.run(["python3", concatenation_script_path,
                    "--folder", evaluation_folder,
                    "--output", batch_run_folder,
                    "--save-name", "concatened-evaluations"])
    report_path = os.path.join(batch_run_folder, "concatened-evaluations.json")
    
    print(f"Evaluation report can be found at: {report_path}")