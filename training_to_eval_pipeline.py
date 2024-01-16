import subprocess
import argparse
import json
import os
import itertools

if __name__ == "__main__":
    # We want to register each batch run (this script)
    # We want to register each run (llm training to eval)
    # The name of the trained llm should be representative of the hyperparameters
    # Training an LLM cost, so if one is already trained with the set of hyperparameters we should not train it again
    
    parser = argparse.ArgumentParser(prog="Batch run LLM training to LLM evaluation on SPARQL Dataset",
                                     description="Orchestrate the run of multiple script to train an LLM and evaluate it.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the (json) config file.")
    parser.add_argument("-i", "--id", type=str, required=True, help="ID of the batch run.")
    parser.add_argument("-o", "--output", type=str, help="Where the batch run should save results.", default="./outputs/batch_run/")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"The mandatory config file was not found at: {args.config} .")
    
    os.makedirs(args.output, exist_ok=True)
    
    batch_run_folder = os.path.join(args.output, args.id)
    generation_folder = os.path.join(batch_run_folder, "generation")
    execution_folder = os.path.join(batch_run_folder, "execution")
    evaluation_folder = os.path.join(batch_run_folder, "evaluation")
    
    if os.path.exists(batch_run_folder):
        raise Exception(f"A previous batch run has been executed with this id: {args.id} .")
    os.makedirs(batch_run_folder)
    os.makedirs(generation_folder)
    os.makedirs(execution_folder)
    os.makedirs(evaluation_folder)
    
    config = json.load(open(args.config, "r"))
    
    training_hyperparameters = [
        config['models_to_train'],
        config['training-hyperparameters']["lora-r-value"],
        config['training-hyperparameters']["batch-size"],
        config['training-hyperparameters']["packing"],
    ]
    
    for model_obj, rvalue, batch_size, packing in itertools.product(*training_hyperparameters):
        # 1) Train an LLM (sft_peft.py)
        print(f"Starting LLM Training: {model_obj['name']=}, {rvalue=}, {batch_size=}, {bool(packing)=}")

        full_model_name = f"{model_obj['name']}_{config['training-hyperparameters-name-abbreviation']['lora-r-value']}{rvalue}-{config['training-hyperparameters-name-abbreviation']['batch-size']}{batch_size}-{config['training-hyperparameters-name-abbreviation']['packing']}{packing}"
        
        training_return = subprocess.run(["accelerate", "launch", "scripts/sft_peft.py",
                                          "--model", model_obj['path'],
                                          "--train-data", config["datasets"]["train"],
                                          "--test-data", config["datasets"]["test"],
                                          "--valid-data", config["datasets"]["valid"],
                                          "--rvalue", str(rvalue),
                                          "--batch-size", str(batch_size),
                                          "--gradient-accumulation", str(4),
                                          "--packing", str(packing),
                                          "--output", args.output,
                                          "--save-name", full_model_name,
                                          "--save-adapters",
                                          "--save-merged"
                                          ])
        
        
        if training_return.returncode != 0:
            print(f"Failed to train: {full_model_name}.")
            continue
        
        merged_model_path = os.path.join([args.output, full_model_name])
        
        # 2) Generate sparql queries using libwikidatallm
        print(f"Generating SPARQL queries: model={full_model_name}, temperature={config['evaluation-hyperparameters']['temperature']}, top-p={config['evaluation-hyperparameters']['top-p']}")
        
        generation_name = f"{full_model_name}_{config['evaluation-hyperparameters-name-abbreviation']['temperature']}{config['evaluation-hyperparameters']['temperature']}-{config['evaluation-hyperparameters-name-abbreviation']['top-p']}{config['evaluation-hyperparameters']['top-p']}"
        generate_queries_return = subprocess.run(["python3", "-m", "libwikidatallm",
                                                  "--test-data", config["datasets"]["test"],
                                                  "--model", merged_model_path,
                                                  "--tokenizer", model_obj['path'],
                                                  "--context-length", model_obj['context-length'],
                                                  "--temperature", str(config['evaluation-hyperparameters']['temperature']),
                                                  "--topp", str(config['evaluation-hyperparameters']['top-p']),
                                                  "--num-tokens", str(256),
                                                  "--output", generation_folder,
                                                  "--save-name", generation_name
                                                  ])
        
        if generate_queries_return.returncode != 0:
            print(f"Failed to generate queries: {generation_name}.")
            continue
        
        generated_queries_path = os.path.join([generation_folder, f"{generation_name}.parquet.gzip"])
        
        # 3) Execute queries on wikidata (execute_queries.py)
        print("Executing generated queries on Wikidata's SPARQL Endpoint.")
        execute_name = f"{generation_name}_executed"
        execute_queries_return = subprocess.run(["python3", "execute_queries.py",
                                                 "--dataset", generated_queries_path,
                                                 "--column-name", "translated_prompt",
                                                 "--timeout", str(60),
                                                 "--limit", str(10),
                                                 "--output", execution_folder,
                                                 "--save-name", execute_name,
                                                 ])
        
        if execute_queries_return.returncode != 0:
            print(f"Failed to execute queries: {execute_name}.")
            continue
        
        executed_queries_path = os.path.join([execution_folder, f"{execute_name}.parquet.gzip"])
        
        # 4) Evaluate the LLM
        print(f"Evaluating {full_model_name}.")
        
        evaluate_name = f"{execute_name}_evaluated"
        evaluate_return = subprocess.run(["python3", "script/evaluation_bench.py",
                                          "--dataset", executed_queries_path,
                                          "--model", full_model_name,
                                          "--output", evaluation_folder,
                                          "--save-name", evaluate_name])

        if evaluate_return.returncode != 0:
            print(f"Failed to evaluate llm: {evaluate_name}.")
            continue
        
        # TODO: remove that it's for testing purposes
        break
    
    print("Concatenating all evaluations.")
    subprocess.run(["python3", "scripts/concatenate_evaluations.py",
                    "--folder", evaluation_folder,
                    "--output", batch_run_folder,
                    "--save-name", "concatened-evaluations"])
    report_path = os.path.join(batch_run_folder, "concatened-evaluations.json")
    
    print(f"Evaluation report can be found at: {report_path}")