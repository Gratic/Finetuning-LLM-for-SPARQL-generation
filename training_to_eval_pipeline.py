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
    
    batch_run_result = os.path.join(args.output, args.id)
    
    if os.path.exists(batch_run_result):
        raise Exception(f"A previous batch run has been executed with this id: {args.id} .")
    os.makedirs(batch_run_result)
    
    config = json.load(open(args.config, "r"))
    
    training_hyperparameters = [
        config['models_to_train'],
        config['training-hyperparameters']["lora-r-value"],
        config['training-hyperparameters']["batch-size"],
        config['training-hyperparameters']["packing"],
    ]
    
    for model_obj, rvalue, batch_size, packing in itertools.product(*training_hyperparameters):
        # 1) Train an LLM (sft_peft.py)
        print(f"Starting LLM Training: {model_obj['name']=}, {rvalue=},{batch_size=}, {bool(packing)=}")

        save_name = f"{model_obj['name']}_{config['training-hyperparameters-name-abbreviation']['lora-r-value']}{rvalue}-{config['training-hyperparameters-name-abbreviation']['batch-size']}{batch_size}-{config['training-hyperparameters-name-abbreviation']['packing']}{packing}"
        training_return = subprocess.run(["accelerate", "launch", "script/sft_peft.py",
                                          "--model", model_obj['path'],
                                          "--train-data", config["datasets"]["train"],
                                          "--test-data", config["datasets"]["test"],
                                          "--valid-data", config["datasets"]["valid"],
                                          "--rvalue", rvalue,
                                          "--batch-size", batch_size,
                                          "--gradient-accumulation", 4,
                                          "--packing", packing,
                                          "--output", args.output,
                                          "--save-name", save_name,
                                          "--save-adapters",
                                          "--save-merged"
                                          ])
        
        if training_return.returncode != 0:
            print(f"Failed to train: {save_name}.")
            continue
        
        merged_model_path = os.path.join([args.output, save_name])
        
        # 2) Generate sparql queries using libwikidatallm
        generation_name = f"{save_name}_{config['evaluation-hyperparameters-name-abbreviation']['temperature']}{config['evaluation-hyperparameters']['temperature']}-{config['evaluation-hyperparameters-name-abbreviation']['top-p']}{config['evaluation-hyperparameters']['top-p']}"
        generate_queries_return = subprocess.run(["python3", "-m", "libwikidatallm",
                                                  "--test-data", config["datasets"]["test"],
                                                  "--model", merged_model_path,
                                                  "--tokenizer", model_obj['path'],
                                                  "--context-length", model_obj['context-length'],
                                                  "--temperature", config['evaluation-hyperparameters']['temperature'],
                                                  "--topp", config['evaluation-hyperparameters']['top-p'],
                                                  "--num-tokens", 256,
                                                  "--output", batch_run_result,
                                                  "--save-name", generation_name
                                                  ])
        
        generated_queries_path = os.path.join([args.output, f"{generation_name}.parquet.gzip"])
        
    # 3) Execute queries on wikidata (execute_queries.py)
    # 4) Evaluate the LLM