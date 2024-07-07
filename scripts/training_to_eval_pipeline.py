import argparse
import itertools
import json
import logging
import os
import subprocess
from typing import Dict, List
import configparser
from pathlib import Path

TRAINING_SCRIPT = "scripts/sft_peft.py"
MERGING_SCRIPT = "scripts/merge_adapters.py"
LIBWIKIDATALLM_PATH = "modules.libwikidatallm"
EXECUTING_QUERIES_SCRIPT = "scripts/execute_queries.py"
EVALUATION_SCRIPT = "scripts/evaluation_bench.py"
CONCATENATION_SCRIPT = "scripts/concatenate_evaluations.py"

def file_exists_or_raise(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file was not found in: {file_path}")

def generate_name_from_dict(params_dict: Dict, abbrev_dict: Dict):
    opt_abrev = {
        "adamw_torch": "awtor",
        "adamw_bnb_8bit": "awbn8"
    }
    return "-".join([abbrev_dict[key] + keep_only_alphanum_chars(str(opt_abrev.get(params_dict[key], params_dict[key]))) for key in filter(lambda x: abbrev_dict[x] != "no", list(params_dict.keys()))])

def generate_folder_structure(args):
    os.makedirs(args.output, exist_ok=True)
    
    batch_run_folder = os.path.join(args.output, args.id)
    generation_folder = os.path.join(batch_run_folder, "generation")
    execution_folder = os.path.join(batch_run_folder, "execution")
    evaluation_folder = os.path.join(batch_run_folder, "evaluation")
    models_folder = os.path.join(batch_run_folder, "models")
    
    if os.path.exists(batch_run_folder) and not args.recover:
        raise Exception(f"A previous batch run has been executed with this id: {args.id} .")
    
    os.makedirs(batch_run_folder, exist_ok=args.recover)
    os.makedirs(generation_folder, exist_ok=args.recover)
    os.makedirs(execution_folder, exist_ok=args.recover)
    os.makedirs(evaluation_folder, exist_ok=args.recover)
    os.makedirs(models_folder, exist_ok=args.recover)
    return batch_run_folder, generation_folder, execution_folder, evaluation_folder, models_folder

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

def copy_config_file_into_batch_run_folder(config_path, batch_run_folder):
    config = Path(config_path).read_text()
    Path(batch_run_folder).joinpath("config.ini").write_text(config)

def validate_config(config: configparser.ConfigParser):
    if not config["Datasets"].get("hf_dataset", None):
        raise ValueError("The training needs a huggingface dataset")

def get_training_hyperparameters(config: configparser.ConfigParser) -> List:
    return [
        json.loads(config['Models'].get('models')),
        config['Training Hyperparameters'].getlist("optimizer"),
        config['Training Hyperparameters'].getlist("learning_rate"),
        config['Training Hyperparameters'].getlist("lora_r_value"),
        config['Training Hyperparameters'].getlist("lora_r_alpha_mult"),
        config['Training Hyperparameters'].getlist("lora_dropout"),
        config['Training Hyperparameters'].getlist("batch_size"),
        config['Training Hyperparameters'].getlist("packing"),
        config['Training Hyperparameters'].getlist("neft_tune_alpha"),
        config['Training Hyperparameters'].getlist("gradient_accumulation"),
        config['Training Hyperparameters'].getlist("gradient_checkpointing"),
        config['Training Hyperparameters'].getlist("pipeline_type"),
        config['Training Hyperparameters'].getlist("input_type"),
    ]
    
def run_subprocess(command: List[str], error_message: str):
    result = subprocess.run(command)
    if result.returncode != 0:
        logging.error(f"{error_message}")
        print(f"{error_message}")
        return False
    return True

def train_model(args: argparse.Namespace, config: configparser.ConfigParser, model_obj: Dict, train_params: Dict, 
                pipeline_type: str, input_type: str, models_folder: str, log_file: str) -> str:
    full_model_name = f"{model_obj['name']}_{generate_name_from_dict(train_params, config['Training Hyperparameters Name Abbreviations'])}-ctx{model_obj['context_length']}-q{model_obj.get('quantization', '4bit')}-{pipeline_type}-{input_type}-st{keep_only_alphanum_chars(config['Evaluation Hyperparameters']['start_tag'])}"
    
    adapters_model_path = os.path.join(models_folder, f"{full_model_name}_adapters")
    
    if os.path.exists(adapters_model_path):
        logging.info(f"Recovered adapter for: {full_model_name}")
        return adapters_model_path

    
    use_accelerate = config["Execution"].getboolean("use_accelerate")
    message = f"""Starting training:
MODEL INFORMATIONS
Model name: {model_obj['name']}
Model path: {model_obj['path']}
Context Lenght: {model_obj['context_length']}
Quantization: {model_obj.get("quantization", "4bit")}
Token: {model_obj.get("token", "No")}

TRAINING INFORMATIONS
Using Accelerate: {'YES' if use_accelerate else 'NO'}
Optimizer: {train_params['optimizer']}
Computational Datatype: {train_params['computational_datatype']}
Learning Rate: {train_params['learning_rate']}
Number of epochs: {train_params['num_epochs']}
Batch size: {train_params['batch_size']}
Gradient Accumulation: {train_params['gradient_accumulation']}
Gradient Checkpointing: {bool(int(train_params['gradient_checkpointing']))}
Packing: {bool(int(train_params['packing']))}
LoRA -
    rank: {train_params['lora_r_value']}
    alpha: {int(train_params['lora_r_value']) * int(train_params['lora_r_alpha_mult'])} ({train_params['lora_r_alpha_mult']}x)
    dropout: {train_params['lora_dropout']}
Neft Tune Alpha: {train_params['neft_tune_alpha']}
Input type: {input_type}
Pipeline type: {pipeline_type}"""
    print(message)
    
    training_args = (["accelerate", "launch"] if use_accelerate else ["python3"]) + [
        TRAINING_SCRIPT,
        "--model", model_obj['path'],
        "--optimizer", train_params["optimizer"],
        "--computational-datatype", train_params["computational_datatype"],
        "--learning-rate", str(train_params["learning_rate"]),
        "--context-length", str(model_obj['context_length']),
        "--model-quant", model_obj.get("quantization", "4bit"),
        "--target-column", config["Pipeline Types To Target Columns"][pipeline_type],
        "--input-column", config["Input Types to Input Columns"][input_type],
        "--hf-dataset", config["Datasets"].get("hf_dataset", ""),
        "--start-tag", str(config['Evaluation Hyperparameters']['start_tag']),
        "--end-tag", str(config['Evaluation Hyperparameters']['end_tag']),
        "--rvalue", str(train_params["lora_r_value"]),
        "--ralpha", str(train_params["lora_r_alpha_mult"]),
        "--lora-dropout", str(train_params["lora_dropout"]),
        "--batch-size", str(train_params["batch_size"]),
        "--gradient-accumulation", str(train_params["gradient_accumulation"]),
        "--gradient-checkpointing", str(train_params["gradient_checkpointing"]),
        "--packing", str(train_params["packing"]),
        "--neft-tune-alpha", str(train_params["neft_tune_alpha"]),
        "--epochs", str(train_params["num_epochs"]),
        "--output", models_folder,
        "--save-name", full_model_name,
        "--run-name", f"{full_model_name}",
        "--wnb-project", args.id,
        "--save-adapters",
        "--log-level", args.log_level,
        "--log-file", log_file,
        "--random-seed", str(config['Execution'].getint('random_seed')),
        "--token", model_obj.get("token", ""),
    ]
    
    if not config['Execution'].getboolean('do_eval', True):
        training_args.append("--no-eval")

    if not run_subprocess(training_args, f"Failed to train: {full_model_name}."):
        return ""

    return adapters_model_path
    
def generate_queries(config: configparser.ConfigParser, model_obj: Dict, adapters_model_path: str, 
                     pipeline_type: str, input_type: str, generation_folder: str, full_model_name: str) -> str:
    generation_name = f"{full_model_name}_{generate_name_from_dict(config['Evaluation Hyperparameters'], config['Evaluation Hyperparameters Name Abbreviations'])}"
    generated_queries_path = os.path.join(generation_folder, f"{generation_name}.parquet.gzip")
    
    if os.path.exists(generated_queries_path):
        logging.info(f"Recovered generated queries for: {generation_name}")
        return generated_queries_path

    message = f"""Starting generation process
MODEL INFORMATIONS
Model name: {model_obj['name']}
Model path: {model_obj['path']}
Adapter path: {adapters_model_path}
Tokenizer: {model_obj['path']}
Context Lenght: {model_obj['context_length']}

GENERATION INFORMATIONS
Computational Type: {config["Evaluation Hyperparameters"]["computational_type"]}
Column name: {config["Input Types to Input Columns"][input_type]}
Engine: {config["Evaluation Hyperparameters"]["engine"]}
Pipeline: {pipeline_type}
Start tag: {config['Evaluation Hyperparameters']['start_tag']}
End tag: {config['Evaluation Hyperparameters']['end_tag']}
Max new tokens: {str(config['Evaluation Hyperparameters']['num_tokens'])}
"""
    if config["Evaluation Hyperparameters"]["decoding"] == 'sampling':
        message += f"""Decoding: Sampling
Temperature: {config['Evaluation Hyperparameters']['temperature']}
Top p: {config['Evaluation Hyperparameters']['top_p']}"""
    elif config["Evaluation Hyperparameters"]["decoding"] == "greedy":
        message += f"""Decoding: Greedy"""
    print(message)
    
    generate_queries_args = [
        "python3", "-m", LIBWIKIDATALLM_PATH,
        "--data", config["Datasets"]["test"] if config["Datasets"].get("hf_dataset", "") == "" else "",
        "--huggingface_dataset", config["Datasets"].get("hf_dataset", ""),
        "--huggingface_split", "test" if config["Datasets"].get("hf_dataset", "") == "" else "",
        "--column-name", config["Input Types to Input Columns"][input_type],
        "--model", model_obj['path'],
        "--adapters", adapters_model_path,
        "--tokenizer", model_obj['path'],
        "--context-length", str(model_obj['context_length']),
        "--engine", config["Evaluation Hyperparameters"]["engine"],
        "--computational-type", config["Evaluation Hyperparameters"]["computational_type"],
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
        "--random-seed", str(config['Execution'].getint('random_seed')),
        "--token", model_obj.get("token", ""),
    ]

    if not run_subprocess(generate_queries_args, f"Failed to generate queries: {generation_name}."):
        return ""

    if not os.path.exists(generated_queries_path):
        raise FileNotFoundError(f"The generated queries were not found: {generated_queries_path}.")

    return generated_queries_path

def execute_queries(generated_queries_path: str, execution_folder: str) -> str:
    execute_name = f"{Path(generated_queries_path).stem}_executed"
    executed_queries_path = os.path.join(execution_folder, f"{execute_name}.parquet.gzip")
    
    if os.path.exists(executed_queries_path):
        logging.info(f"Recovered execution: {execute_name}")
        return executed_queries_path

    execute_queries_args = [
        "python3", EXECUTING_QUERIES_SCRIPT,
        "--dataset", generated_queries_path,
        "--column-name", "output",
        "--timeout", str(60),
        "--limit", str(10),
        "--output", execution_folder,
        "--save-name", execute_name,
    ]

    if not run_subprocess(execute_queries_args, f"Failed to execute queries: {execute_name}."):
        return ""

    if not os.path.exists(executed_queries_path):
        raise FileNotFoundError(f"The executed queries were not found: {executed_queries_path}.")

    return executed_queries_path

def evaluate_model(executed_queries_path: str, full_model_name: str, config: configparser.ConfigParser, pipeline_type: str, 
                   evaluation_folder: str, log_file: str, log_level: str) -> bool:
    evaluate_name = f"{Path(executed_queries_path).stem}_evaluated"
    evaluate_args = [
        "python3", EVALUATION_SCRIPT,
        "--dataset", executed_queries_path,
        "--generated-field", "output",
        "--executed-field", "execution",
        "--hf-dataset", config["Datasets"].get("hf_dataset", ""),
        "--hf-split", "test",
        "--hf-target", config["Pipeline Types To Target Columns"][pipeline_type],
        "--model", full_model_name,
        "--output", evaluation_folder,
        "--save-name", evaluate_name,
        "--log-level", log_level,
        "--log-file", log_file
    ]

    return run_subprocess(evaluate_args, f"Failed to evaluate llm: {evaluate_name}.")

def concatenate_evaluations(evaluation_folder: str, batch_run_folder: str):
    logging.info("Concatenating all evaluations.")
    print("Concatenating all evaluations.")
    
    concatenation_args = [
        "python3", CONCATENATION_SCRIPT,
        "--folder", evaluation_folder,
        "--output", batch_run_folder,
        "--save-name", "concatened-evaluations"
    ]
    
    if run_subprocess(concatenation_args, "Failed to concatenate evaluations."):
        report_path = os.path.join(batch_run_folder, "concatened-evaluations.json")
        print(f"Evaluation report can be found at: {report_path}")

def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="Batch run LLM training to LLM evaluation on SPARQL Dataset",
        description="Orchestrate the run of multiple scripts to train an LLM and evaluate it."
    )
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the (ini) config file.")
    parser.add_argument("-i", "--id", type=str, required=True, help="ID of the batch run.")
    parser.add_argument("-o", "--output", type=str, help="Where the batch run should save results.", default="./outputs/batch_run/")
    parser.add_argument("-log", "--log-level", type=str, help="Logging level (debug, info, warning, error, critical).", default="warning")
    parser.add_argument("-r", "--recover", action="store_true", help="Try to recover a failed run from the id.")
    parser.add_argument("-t", "--training-only", action="store_true", help="Only train the models, no testing.")
    return parser.parse_args()

def main(args :argparse.Namespace):
    """
    This script is an "orchestrator". It takes a config file, do some preprocessing (gold/test dataset execution) and then the training of multiple LLMs from it.
    If the training is succesful (sft_peft.py returns 0) it will:
    - generate queries using it,
    - execute the generated queries on wikidata's SPARQL endpoint,
    - evaluate it.
    When all evaluations are dones, the script launches a last script that takes all the evaluations and concatenates it in one file.
    """
    
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"The mandatory config file was not found at: {args.config} .")
    
    map(file_exists_or_raise, [ 
        TRAINING_SCRIPT,
        MERGING_SCRIPT,
        LIBWIKIDATALLM_PATH,
        EXECUTING_QUERIES_SCRIPT,
        EVALUATION_SCRIPT,
        CONCATENATION_SCRIPT
        ])
    
    batch_run_folder, generation_folder, execution_folder, evaluation_folder, models_folder = generate_folder_structure(args)
    
    copy_config_file_into_batch_run_folder(args.config, batch_run_folder)
    
    log_file = setup_logging(args, batch_run_folder)
    
    logging.info("Loading config dataset.")
    config = configparser.ConfigParser(allow_no_value=True, converters={"list": lambda x: [i.strip() for i in x.split(',')]})
    config.read(args.config)
    
    random_seed = config['Execution'].getint('random_seed')
    
    # 0.1) Execute the test dataset against wikidata API
    validate_config(config)
    
    training_hyperparameters = get_training_hyperparameters(config)
    
    num_epochs = config['Training Hyperparameters'].getint("epochs")
    
    config['Evaluation Hyperparameters']['start_tag'] = config['Evaluation Hyperparameters'].get('start_tag').replace('\\n', '\n')
    config['Evaluation Hyperparameters']['end_tag'] = config['Evaluation Hyperparameters'].get('end_tag').replace('\\n', '\n')
    
    logging.info("Starting the training and evaluation loop.")
    for params in itertools.product(*training_hyperparameters):
        model_obj, optimizer, learning_rate, rvalue, ralphamult, lora_dropout, batch_size, packing, neft_tune_alpha, gradient_accumulation, gradient_checkpointing, pipeline_type, input_type = params
        # 1) Train an LLM (sft_peft.py)
        packing = int(packing)

        training_computational_datatype = config["Training Hyperparameters"].get("computational_datatype", 'fp16')

        train_params = {
            "optimizer": optimizer,
            "computational_datatype": training_computational_datatype,
            "learning_rate": learning_rate,
            "lora_r_value": rvalue,
            "lora_r_alpha_mult": ralphamult,
            "lora_dropout": lora_dropout,
            "batch_size": batch_size,
            "gradient_accumulation": gradient_accumulation,
            "gradient_checkpointing": gradient_checkpointing,
            "packing": packing,
            "neft_tune_alpha": neft_tune_alpha,
            "num_epochs": num_epochs
        }
        
        adapters_model_path = train_model(args, config, model_obj, train_params, pipeline_type, input_type, models_folder, log_file)
        if not adapters_model_path:
            continue
        
        if args.training_only:
            continue
        
        # 2) Generate sparql queries using libwikidatallm
        generated_queries_path = generate_queries(config, model_obj, adapters_model_path, pipeline_type, input_type, generation_folder, Path(adapters_model_path).stem)
        if not generated_queries_path:
            continue
            
        # 3) Execute queries on wikidata (execute_queries.py)        
        executed_queries_path = execute_queries(generated_queries_path, execution_folder)
        if not executed_queries_path:
            continue
            
        # 4) Evaluate the LLM
        if not evaluate_model(executed_queries_path, Path(adapters_model_path).stem, config, pipeline_type, evaluation_folder, log_file, args.log_level):
            continue
        
    if not args.training_only:
        concatenate_evaluations(evaluation_folder, batch_run_folder)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)