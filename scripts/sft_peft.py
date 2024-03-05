import sys
from pathlib import Path
sys.path.append(Path("modules").absolute().__str__())

from datasets import load_dataset
from evaluation_utils import is_correct_SPARQL_query
from execution_utils import is_query_empty, can_add_limit_clause, add_relevant_prefixes_to_query, send_query_to_api
from data_utils import get_nested_values, safe_eval, set_seed
from evaluation_utils import compute_precision, compute_recall
from peft import LoraConfig
from transformers import TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import argparse
import evaluate
import logging
import nltk
import numpy as np
import os
import torch

tokenizer = None
rouge_metric = None
target_column = None

# https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/supervised_finetuning.py
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def format_prompt_packing(example):
    text = f"[INST] Given a question, generate a SPARQL query that answers the question where entities and properties are placeholders. After the generated query, gives the list of placeholders and their corresponding Wikidata identifiers: {example['input']} [/INST] `sparql\n{example[target_column]}`"
    return text

def format_prompt(example):
    output_texts = []
    for i in range(len(example['input'])):
        text = f"[INST] Given a question, generate a SPARQL query that answers the question where entities and properties are placeholders. After the generated query, gives the list of placeholders and their corresponding Wikidata identifiers: {example['input'][i][0]} [/INST] `sparql\n{example[target_column][i]}`"
        output_texts.append(text)
    return output_texts

def parse_args():
    parser = argparse.ArgumentParser(prog="PEFT (QLora) SFT Script")
    parser.add_argument("-m", "--model", type=str, help="Huggingface model or path to a model to finetune.", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("-trd", "--train-data", required=True, type=str, help="Path to the train dataset.")
    parser.add_argument("-trg", "--target-column", type=str, help="Indicates which column to use for answers (default= 'target_template').", default="target_template")
    parser.add_argument("-vd", "--valid-data", required=False, type=str, help="Path to the valid dataset.", default="")
    parser.add_argument("-rv", "--rvalue", type=int, help="Lora r-value.", default=8)
    parser.add_argument("-ld", "--lora-dropout", type=float, help="Lora dropout value.", default=0.05)
    parser.add_argument("-bs", "--batch-size", type=int, help="Batch size for training.", default=1)
    parser.add_argument("-ga", "--gradient-accumulation", type=int, help="Gradient accumulation, number of batch to process before making an optimizer step.", default=4)
    parser.add_argument("-p", "--packing", type=int, help="Train with Packing or not (1=True, 0=False).",  default=0)
    parser.add_argument("-nta", "--neft-tune-alpha", type=int, help="A different value from 0. will use Neft Tuning.",  default=0)
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs to train for.",  default=3)
    parser.add_argument("-o", "--output", type=str, help="Output directory", default="")
    parser.add_argument("-sn", "--save-name", type=str, help="The folder name where the saved checkpoint will be found.", default="final_checkpoint")
    parser.add_argument("-sa", "--save-adapters", dest='save_adapters', action='store_true', help="Save the adapters.")
    parser.add_argument("-wp", "--wnb-project", type=str, help="Weight and Biases project name.", default="SFT_Training test")
    parser.add_argument("-rn", "--run-name", type=str, help="Weight and Biases name of the run.", default=None)
    parser.add_argument("-wl", "--wnb-log", type=str, help="Weight and Biases log model.", default="checkpoint")
    parser.add_argument("-log", "--log-level", type=str, help="Logging level (debug, info, warning, error, critical).", default="warning")
    parser.add_argument("-logf", "--log-file", type=str, help="Logging file.", default="")
    parser.add_argument("-acc", "--accelerate", help="Use accelerate.", action="store_true")
    parser.add_argument("-rand", "--random-seed", type=int, help="Set up a random seed if specified.", default=0)
    args = parser.parse_args()
    return args

# https://github.com/huggingface/trl/issues/862#issuecomment-1896074498
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1) # Greedy decoding

def extract_query(query):
    if query.find('`sparql') != -1 and query.rfind('`') != -1:
        start_sparql = query.find('`sparql')
        end_sparql = query.rfind('`')
        
        return query[start_sparql+8:end_sparql]
    return None

def execute_query(query):
    query = extract_query(query)
    if is_query_empty(query):
        return None
    else:
        query = add_relevant_prefixes_to_query(query)
        
        if can_add_limit_clause(query):
            query += f"\nLIMIT 10"
    
    response = send_query_to_api(query, do_print=False)
    
    if isinstance(response, str):
        if response.startswith(('exception:', 'timeout')):
            return None
        
        response = safe_eval(response)
        
        if response == None:
            return None
            
    return response

# https://huggingface.co/docs/evaluate/transformers_integrations
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    
    # decode preds and labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id) # replace -100 in labels with the padding token
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    batch_size = len(decoded_preds)
    
    # rougeLSum expects newline after each sentence
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]

    executed_labels = [execute_query(query) for query in decoded_labels]
    executed_preds = [execute_query(query) for query in decoded_preds]
    
    # Correct query computation
    filtered_queries = list(filter(lambda x: x[0] != None and x[1] != None, zip(executed_labels, executed_preds)))
    nested_values = list(map(lambda x: (get_nested_values(x[0]), get_nested_values(x[1])), filtered_queries))
    
    precc = sum([compute_precision(hyp, gold) for hyp, gold in nested_values] if len(nested_values) > 0 else [0])/batch_size
    recall = sum([compute_recall(hyp, gold) for hyp, gold in nested_values]  if len(nested_values) > 0 else [0])/batch_size

    results_dict = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    correct_syntax = float(sum([is_correct_SPARQL_query(query) for query in decoded_preds]))/batch_size
 
    results_dict.update({"correct_syntax": correct_syntax, "precision": precc, "recall": recall})
    return results_dict

def main():
    global tokenizer, rouge_metric, target_column
    
    args = parse_args()
    
    setup_logging(args)
    
    if args.random_seed != 0:
        set_seed(args.random_seed)
    
    datafiles = {
            "train": args.train_data
        }
    
    has_valid_dataset = False
    if args.valid_data != None and args.valid_data != "":
        datafiles.update({"valid": args.valid_data})
        has_valid_dataset = True
    
    save_path_adapters = os.path.join(args.output, f"{args.save_name}_adapters")
    
    do_packing = bool(args.packing)
    
    logging.info("Loading datasets.")
    print("Loading datasets.")
    dataset = load_dataset("pandas", data_files=datafiles)
    
    if args.target_column not in dataset.column_names['train']:
        raise ValueError(f"The target column was not found in the test dataset, have: {args.target_column}, found: {dataset.column_names['test']}.")
    
    if has_valid_dataset and args.target_column not in dataset.column_names['valid']:
        raise ValueError(f"The target column was not found in the valid dataset, have: {args.target_column}, found: {dataset.column_names['valid']}.")
    
    target_column = args.target_column
    
    model_id = args.model
    
    os.environ["WANDB_PROJECT"] = args.wnb_project  # name your W&B project
    os.environ["WANDB_LOG_MODEL"] = args.wnb_log  # log all model checkpoints

    lora_config = LoraConfig(
        r=args.rvalue,
        lora_alpha=args.rvalue*2,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    logging.info(f"Loading model: {model_id}.")
    print(f"Loading model: {model_id}.")
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto" # torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    
    logging.info(f"Loading tokenizer: {model_id}.")
    print(f"Loading tokenizer: {model_id}.")
    # https://medium.com/@parikshitsaikia1619/mistral-mastery-fine-tuning-fast-inference-guide-62e163198b06
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"
    # TODO: Create a padding token
    tokenizer.pad_token = tokenizer.unk_token
    pretrained_model.config.pad_token_id = tokenizer.pad_token_id

    print_trainable_parameters(pretrained_model)
    
    nltk.download("punkt", quiet=True)
    rouge_metric = evaluate.load("rouge")
    
    training_args = TrainingArguments(
        bf16=True,
        output_dir=save_path_adapters,
        optim="adamw_bnb_8bit",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        neftune_noise_alpha=args.neft_tune_alpha if args.neft_tune_alpha != 0 else None,
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
        evaluation_strategy="epoch" if has_valid_dataset else "No",
        num_train_epochs=args.epochs,
        save_strategy="no", # TODO: maybe save checkpoints and do evaluation with them later
        logging_strategy="epoch",
        run_name=args.run_name,
        report_to="wandb",
        seed=args.random_seed
    )

    collator = None if do_packing else DataCollatorForCompletionOnlyLM(response_template="[/INST]", tokenizer=tokenizer)
    trainer = SFTTrainer(
        pretrained_model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"] if has_valid_dataset else None,
        formatting_func= format_prompt_packing if do_packing else format_prompt,
        max_seq_length=4096,
        peft_config=lora_config,
        packing=do_packing,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
        dataset_num_proc=1
    )

    logging.info(f"Starting training.")
    print("Starting training.")
    trainer.train()
        
    if args.save_adapters:
        logging.info(f"Saving adapters.")
        print("Saving adapters.")
        trainer.model.save_pretrained(save_path_adapters)

def setup_logging(args):
    numeric_log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_log_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}.")
    logging.basicConfig(filename=args.log_file if args.log_file else None, level=numeric_log_level)

if __name__ == "__main__":
    main()