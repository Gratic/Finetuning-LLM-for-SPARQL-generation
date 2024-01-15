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
        config['training-hyperparameters']["lora-r-value"],
        config['training-hyperparameters']["batch-size"],
        config['training-hyperparameters']["packing"],
    ]
    
    for rvalue, batch_size, packing in itertools.product(*training_hyperparameters):
        print(f"Starting LLM Training: {rvalue=},{batch_size=}, {packing=}")
        
        
    
    # 1) Train an LLM (sft_peft.py)
    # 1.5) Merge model adapters (merge_adapters.py)
    # 2) Generate sparql queries using libwikidatallm
    # 3) Execute queries on wikidata (execute_queries.py)
    # 4) Evaluate the LLM