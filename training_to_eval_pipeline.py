import argparse
import itertools
import json
import logging
import os
import subprocess

def file_exists_or_raise(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The script was not found in: {file_path}")

if __name__ == "__main__":
    """
    This script is an "orchestrator". It takes a config file and runs the training of multiple LLMs from it.
    If the training is succesful (sft_peft.py returns 0) it will:
    - generate queries using it,
    - execute the generated queries on wikidata's SPARQL endpoint,
    - evaluate it.
    
    When all evaluations are dones, the script launches a last script that takes all the evaluations and concatenates it in one file.
    """
    
    parser = argparse.ArgumentParser(prog="Batch run LLM training to LLM evaluation on SPARQL Dataset",
                                     description="Orchestrate the run of multiple script to train an LLM and evaluate it.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the (json) config file.")
    parser.add_argument("-i", "--id", type=str, required=True, help="ID of the batch run.")
    parser.add_argument("-g", "--gold", type=str, required=True, help="The gold dataset to compare results to.")
    parser.add_argument("-o", "--output", type=str, help="Where the batch run should save results.", default="./outputs/batch_run/")
    parser.add_argument("-log", "--log-level", type=str, help="Logging level (debug, info, warning, error, critical).", default="warning")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"The mandatory config file was not found at: {args.config} .")
    if not os.path.exists(args.gold):
        raise FileNotFoundError(f"The mandatory gold dataset was not found at: {args.gold} .")
    
    training_script_path = "scripts/sft_peft.py"
    merging_script_path = "scripts/merge_adapters.py"
    libwikidatallm_path = "libwikidatallm"
    executing_queries_script_path = "execute_queries.py"
    evaluation_script_path = "scripts/evaluation_bench.py"
    concatenation_script_path = "scripts/concatenate_evaluations.py"
    
    map(file_exists_or_raise, [training_script_path, merging_script_path, libwikidatallm_path, executing_queries_script_path, evaluation_script_path, concatenation_script_path])
    
    os.makedirs(args.output, exist_ok=True)
    
    batch_run_folder = os.path.join(args.output, args.id)
    generation_folder = os.path.join(batch_run_folder, "generation")
    execution_folder = os.path.join(batch_run_folder, "execution")
    evaluation_folder = os.path.join(batch_run_folder, "evaluation")
    
    # if os.path.exists(batch_run_folder):
    #     raise Exception(f"A previous batch run has been executed with this id: {args.id} .")
    
    os.makedirs(batch_run_folder, exist_ok=True)
    os.makedirs(generation_folder, exist_ok=True)
    os.makedirs(execution_folder, exist_ok=True)
    os.makedirs(evaluation_folder, exist_ok=True)
    
    numeric_log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_log_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}.")
    
    log_file = os.path.join(batch_run_folder, "outputs.log")
    logging.basicConfig(filename=log_file, level=numeric_log_level)
    
    logging.info("Loading config dataset.")
    config = json.load(open(args.config, "r"))
    
    training_hyperparameters = [
        config['models_to_train'],
        config['training-hyperparameters']["lora-r-value"],
        config['training-hyperparameters']["batch-size"],
        config['training-hyperparameters']["packing"],
    ]
    
    logging.info("Starting the training and evaluation loop.")
    for model_obj, rvalue, batch_size, packing in itertools.product(*training_hyperparameters):
        # # 1) Train an LLM (sft_peft.py)
        # logging.info(f"Starting LLM Training: {model_obj['name']=}, {rvalue=}, {batch_size=}, {bool(packing)=}")
        # print(f"Starting LLM Training: {model_obj['name']=}, {rvalue=}, {batch_size=}, {bool(packing)=}")

        full_model_name = f"{model_obj['name']}_{config['training-hyperparameters-name-abbreviation']['lora-r-value']}{rvalue}-{config['training-hyperparameters-name-abbreviation']['batch-size']}{batch_size}-{config['training-hyperparameters-name-abbreviation']['packing']}{packing}"
        
        # adapters_model_path = os.path.join(args.output, f"{full_model_name}_adapters")
        # if not os.path.exists(adapters_model_path):
        #     training_return = subprocess.run(["accelerate", "launch", training_script_path,
        #                                     "--model", model_obj['path'],
        #                                     "--train-data", config["datasets"]["train"],
        #                                     "--test-data", config["datasets"]["test"],
        #                                     "--valid-data", config["datasets"]["valid"],
        #                                     "--rvalue", str(rvalue),
        #                                     "--batch-size", str(batch_size),
        #                                     "--gradient-accumulation", str(4),
        #                                     "--packing", str(packing),
        #                                     "--output", args.output,
        #                                     "--save-name", full_model_name,
        #                                     "--save-adapters",
        #                                     "--log-level", args.log_level,
        #                                     "--log-file", log_file
        #                                     ])
            
        #     if training_return.returncode != 0:
        #         logging.error(f"Failed to train: {full_model_name}.")
        #         print(f"Failed to train: {full_model_name}.")
        #         continue
            
        # merged_model_path = os.path.join(args.output, full_model_name)
        
        # if not os.path.exists(merged_model_path):
        #     if not os.path.exists(adapters_model_path):
        #         raise FileNotFoundError(f"The adapters model was not found: {adapters_model_path}.")
            
        #     merging_return = subprocess.run(["python3", merging_script_path,
        #                                      "-m", model_obj['path'],
        #                                      "-a", adapters_model_path,
        #                                      "-o", merged_model_path
        #                                      ])
            
        #     if merging_return.returncode != 0:
        #         logging.error(f"Failed to merge: {full_model_name}.")
        #         print(f"Failed to merge: {full_model_name}.")
        #         continue
        
        # # 2) Generate sparql queries using libwikidatallm
        # logging.info(f"Generating SPARQL queries: model={full_model_name}, temperature={config['evaluation-hyperparameters']['temperature']}, top-p={config['evaluation-hyperparameters']['top-p']}")
        # print(f"Generating SPARQL queries: model={full_model_name}, temperature={config['evaluation-hyperparameters']['temperature']}, top-p={config['evaluation-hyperparameters']['top-p']}")
        
        # generation_name = f"{full_model_name}_{config['evaluation-hyperparameters-name-abbreviation']['engine']}{config['evaluation-hyperparameters']['engine']}-{config['evaluation-hyperparameters-name-abbreviation']['pipeline']}{config['evaluation-hyperparameters']['pipeline']}-{config['evaluation-hyperparameters-name-abbreviation']['temperature']}{config['evaluation-hyperparameters']['temperature']}-{config['evaluation-hyperparameters-name-abbreviation']['top-p']}{config['evaluation-hyperparameters']['top-p']}"
        generation_name = f"{full_model_name}_{config['evaluation-hyperparameters-name-abbreviation']['temperature']}{config['evaluation-hyperparameters']['temperature']}-{config['evaluation-hyperparameters-name-abbreviation']['top-p']}{config['evaluation-hyperparameters']['top-p']}"
        # generate_queries_return = subprocess.run(["python3", "-m", libwikidatallm_path,
        #                                           "--test-data", config["datasets"]["test"],
        #                                           "--model", merged_model_path,
        #                                           "--tokenizer", model_obj['path'],
        #                                           "--context-length", str(model_obj['context-length']),
        #                                           "--engine", config["evaluation-hyperparameters"]["engine"],
        #                                           "--pipeline", config["evaluation-hyperparameters"]["pipeline"],
        #                                           "--temperature", str(config['evaluation-hyperparameters']['temperature']),
        #                                           "--topp", str(config['evaluation-hyperparameters']['top-p']),
        #                                           "--num-tokens", str(256),
        #                                           "--output", generation_folder,
        #                                           "--save-name", generation_name
        #                                           ])
        
        # if generate_queries_return.returncode != 0:
        #     logging.error(f"Failed to generate queries: {generation_name}.")
        #     print(f"Failed to generate queries: {generation_name}.")
        #     continue
        
        # generated_queries_path = os.path.join(generation_folder, f"{generation_name}.parquet.gzip")
        
        # if not os.path.exists(generated_queries_path):
        #     raise FileNotFoundError(f"The generated queries were not found: {generated_queries_path}.")
        
        # # 3) Execute queries on wikidata (execute_queries.py)
        # logging.info("Executing generated queries on Wikidata's SPARQL Endpoint.")
        # print("Executing generated queries on Wikidata's SPARQL Endpoint.")
        
        execute_name = f"{generation_name}_executed"
        # execute_queries_return = subprocess.run(["python3", executing_queries_script_path,
        #                                          "--dataset", generated_queries_path,
        #                                          "--column-name", "translated_prompt",
        #                                          "--timeout", str(60),
        #                                          "--limit", str(10),
        #                                          "--output", execution_folder,
        #                                          "--save-name", execute_name,
        #                                          ])
        
        # if execute_queries_return.returncode != 0:
        #     logging.error(f"Failed to execute queries: {execute_name}.")
        #     print(f"Failed to execute queries: {execute_name}.")
        #     continue
        
        executed_queries_path = os.path.join(execution_folder, f"{execute_name}.parquet.gzip")
        
        # if not os.path.exists(executed_queries_path):
        #     raise FileNotFoundError(f"The executed queries were not found: {executed_queries_path}.")
        
        # 4) Evaluate the LLM
        logging.info(f"Evaluating {full_model_name}.")
        print(f"Evaluating {full_model_name}.")
        
        evaluate_name = f"{execute_name}_evaluated"
        evaluate_return = subprocess.run(["python3", evaluation_script_path,
                                          "--dataset", executed_queries_path,
                                          "--gold", args.gold,
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
        
        # TODO: remove that it's for testing purposes
        logging.critical("The break at the end of the loop has not been removed.")
        break
        
    
    logging.info("Concatenating all evaluations.")
    print("Concatenating all evaluations.")
    
    subprocess.run(["python3", concatenation_script_path,
                    "--folder", evaluation_folder,
                    "--output", batch_run_folder,
                    "--save-name", "concatened-evaluations"])
    report_path = os.path.join(batch_run_folder, "concatened-evaluations.json")
    
    print(f"Evaluation report can be found at: {report_path}")