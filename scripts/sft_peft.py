import sys
from pathlib import Path
sys.path.append(Path("modules").absolute().__str__())

from data_utils import get_nested_values, safe_eval, set_seed, make_dataframe_from_sparql_response
from datasets import load_dataset
from evaluation_utils import is_correct_SPARQL_query, precision_recall_fscore_support_wrapper, cross_product_func, keep_id_columns, average_precision_wrapper
from execution_utils import is_query_empty, can_add_limit_clause, add_relevant_prefixes_to_query, send_query_to_api
from peft import LoraConfig
from prompts_template import PERSONA_BASIC_INSTRUCTION, BASE_MISTRAL_TEMPLATE, BASE_LLAMA_TEMPLATE
from transformers import TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from libwikidatallm.TemplateLLMQuerySender import TemplateLLMQuerySender
from libwikidatallm.Pipeline import OrderedPipeline
from libwikidatallm.EntityExtractor import BracketRegexEntityExtractor
from libwikidatallm.EntityLinker import TakeFirstWikidataEntityLinker
from libwikidatallm.PlaceholderFiller import SimplePlaceholderFiller
from libwikidatallm.PipelineFeeder import SimplePipelineFeeder
import argparse
import evaluate
import logging
import nltk
import numpy as np
import os
import torch
import pandas as pd

tokenizer = None
rouge_metric = None
bleu_metric = None
meteor_metric = None
input_column = None
target_column = None
templater: TemplateLLMQuerySender = None

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

def make_template_pipeline():
    pipeline = OrderedPipeline()
    
    entity_extractor = BracketRegexEntityExtractor(
        input_column='row',
        output_col_entities='extracted_entities',
        output_col_properties='extracted_properties'
        )
    
    # 3. Reverse search the closest values from labels (a possible upgrade for later: Use an LLM to choose the best values)
    entity_linker = TakeFirstWikidataEntityLinker(
        input_column_entities=entity_extractor.output_col_entities,
        input_column_properties=entity_extractor.output_col_properties,
        output_column_entities='linked_entities',
        output_column_properties='linked_properties'
        )
    
    # 4. Replace the labels with the ID we found in 3.
    query_filler = SimplePlaceholderFiller(
        input_column_query='row',
        input_column_entities=entity_linker.output_column_entities,
        input_column_properties=entity_linker.output_column_properties,
        output_column='output'
        )
    
    pipeline.add_step(entity_extractor)
    pipeline.add_step(entity_linker)
    pipeline.add_step(query_filler)

    return pipeline

def execute_pipeline(queries):
    pipeline = make_template_pipeline()
    feeder = SimplePipelineFeeder(pipeline, use_tqdm=False)
    return feeder.process(queries)

def is_query_format_acceptable(query: str):
    query = extract_query(query)
    if is_query_empty(query):
        return False
    return True

def format_prompt_packing(example):
    text = templater.apply_template({
            "system_prompt": PERSONA_BASIC_INSTRUCTION,
            "prompt": example[input_column],
        })
    text += f"`sparql\n{example[target_column]}`"
    return text

def format_prompt(example):
    output_texts = []
    for i in range(len(example[input_column])):
        text = templater.apply_template({
            "system_prompt": PERSONA_BASIC_INSTRUCTION,
            "prompt": example[input_column][i][0],
        })
        text += f"`sparql\n{example[target_column][i]}`"
        output_texts.append(text)

    return output_texts

def parse_args():
    parser = argparse.ArgumentParser(prog="PEFT (QLora) SFT Script")
    parser.add_argument("-m", "--model", type=str, help="Huggingface model or path to a model to finetune.", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("-trd", "--train-data", required=True, type=str, help="Path to the train dataset.")
    parser.add_argument("-trg", "--target-column", type=str, help="Indicates which column to use for answers (default= 'target_template').", default="target_template")
    parser.add_argument("-ic", "--input-column", type=str, help="Indicates which column to use for the input prompt (default= 'basic_input').", default="basic_input")
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

    acceptable_queries = list(filter(lambda x: is_query_format_acceptable(x[0]) and is_query_format_acceptable(x[1]), zip(decoded_labels, decoded_preds)))
    
    if len(acceptable_queries) > 0:
        acceptable_labels = list(map(lambda x: x[0], acceptable_queries))
        acceptable_preds = list(map(lambda x: x[1], acceptable_queries))
        
        translated_labels = list(map(lambda x: x['output'], execute_pipeline(acceptable_labels)))
        translated_preds = list(map(lambda x: x['output'], execute_pipeline(acceptable_preds)))

        executed_labels = [execute_query(query) for query in translated_labels]
        executed_preds = [execute_query(query) for query in translated_preds]
        
        data = pd.DataFrame(data={
            "executed_labels": executed_labels,
            "executed_preds": executed_preds,
        })
        
        data['get_nested_values_labels'] = data.apply(lambda x: get_nested_values(x["executed_labels"]), axis=1)
        data['get_nested_values_preds'] = data.apply(lambda x: get_nested_values(x["executed_preds"]), axis=1)
        
        df_not_null = pd.DataFrame()
        
        df_not_null['labels_df'] = data.apply(lambda x: make_dataframe_from_sparql_response(x['executed_labels']) if isinstance(x['executed_labels'], list) else pd.DataFrame(), axis=1)
        df_not_null['preds_df'] = data.apply(lambda x: make_dataframe_from_sparql_response(x['executed_preds']) if isinstance(x['executed_preds'], list) else pd.DataFrame(), axis=1)
        
        df_not_null['labels_id_columns'] = df_not_null.apply(lambda x: keep_id_columns(x['labels_df']), axis=1)
        df_not_null['preds_id_columns'] = df_not_null.apply(lambda x: keep_id_columns(x['preds_df']), axis=1)
        
        data['get_nested_values_precision_recall_fscore'] = data.apply(lambda x: precision_recall_fscore_support_wrapper(
            x['get_nested_values_labels'],
            x['get_nested_values_preds']
        ), axis=1)
        
        df_not_null['cross_precision_recall_fscore'] = df_not_null.apply(lambda x: cross_product_func(
            func=precision_recall_fscore_support_wrapper,
            y_true=x['labels_df'].apply(lambda y: y.fillna(value="")),
            y_pred=x['preds_df'].apply(lambda y: y.fillna(value="")),
            maximization=True,
            average="samples"
        )
        , axis=1)

        df_not_null['id_precision_recall_fscore'] = df_not_null.apply(lambda x: cross_product_func(
            func=precision_recall_fscore_support_wrapper,
            y_true=x['labels_id_columns'].apply(lambda y: y.fillna(value="")),
            y_pred=x['preds_id_columns'].apply(lambda y: y.fillna(value="")),
            maximization=True,
            average="samples"
        )
        , axis=1)
        
        data['get_nested_values_average_precision'] = data.apply(lambda x: average_precision_wrapper(
            y_true=x['get_nested_values_labels'],
            y_pred=x['get_nested_values_preds']
        ), axis=1)

        df_not_null['cross_average_precision'] = df_not_null.apply(lambda x: cross_product_func(
            func=average_precision_wrapper,
            y_true=x['labels_df'].apply(lambda y: y.fillna(value="")),
            y_pred=x['preds_df'].apply(lambda y: y.fillna(value="")),
            maximization=True,
        )
        , axis=1)

        df_not_null['id_average_precision'] = df_not_null.apply(lambda x: cross_product_func(
            func=average_precision_wrapper,
            y_true=x['labels_id_columns'].apply(lambda y: y.fillna(value="")),
            y_pred=x['preds_id_columns'].apply(lambda y: y.fillna(value="")),
            maximization=True,
        )
        , axis=1)
        
        gnv_precision = data['get_nested_values_precision_recall_fscore'].map(lambda r: r[0] if isinstance(r, tuple) else 0).mean()
        gnv_recall = data['get_nested_values_precision_recall_fscore'].map(lambda r: r[1] if isinstance(r, tuple) else 0).mean()
        gnv_fscore = data['get_nested_values_precision_recall_fscore'].map(lambda r: r[2] if isinstance(r, tuple) else 0).mean()

        cross_precision = df_not_null['cross_precision_recall_fscore'].map(lambda r: r[0] if isinstance(r, tuple) else 0).mean()
        cross_recall = df_not_null['cross_precision_recall_fscore'].map(lambda r: r[1] if isinstance(r, tuple) else 0).mean()
        cross_fscore = df_not_null['cross_precision_recall_fscore'].map(lambda r: r[2] if isinstance(r, tuple) else 0).mean()

        id_precision = df_not_null['id_precision_recall_fscore'].map(lambda r: r[0] if isinstance(r, tuple) else 0).mean()
        id_recall = df_not_null['id_precision_recall_fscore'].map(lambda r: r[1] if isinstance(r, tuple) else 0).mean()
        id_fscore = df_not_null['id_precision_recall_fscore'].map(lambda r: r[2] if isinstance(r, tuple) else 0).mean()

        gnv_map = data['get_nested_values_average_precision'].mean()
        cross_map = df_not_null['cross_average_precision'].mean()
        id_map = df_not_null['id_average_precision'].mean()
    else:
        gnv_precision = 0
        gnv_recall = 0
        gnv_fscore = 0
        cross_precision = 0
        cross_recall = 0
        cross_fscore = 0
        id_precision = 0
        id_recall = 0
        id_fscore = 0
        gnv_map = 0
        cross_map = 0
        id_map = 0
    
    # filtered_queries = list(filter(lambda x: x[0] != None and x[1] != None, zip(executed_labels, executed_preds)))
    # nested_values = list(map(lambda x: (get_nested_values(x[0]), get_nested_values(x[1])), filtered_queries))
    
    # precc = sum([compute_precision(hyp, gold) for hyp, gold in nested_values] if len(nested_values) > 0 else [0])/batch_size
    # recall = sum([compute_recall(hyp, gold) for hyp, gold in nested_values]  if len(nested_values) > 0 else [0])/batch_size

    results_dict = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    bleu_dict = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    meteor_dict = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)

    correct_syntax = float(sum([is_correct_SPARQL_query(query) for _,query in acceptable_queries]))/batch_size
 
    results_dict.update({"correct_syntax": correct_syntax})
    results_dict.update({
        "gnv_precision" : gnv_precision,
        "gnv_recall" : gnv_recall,
        "gnv_fscore" : gnv_fscore,
        "cross_precision" : cross_precision,
        "cross_recall" : cross_recall,
        "cross_fscore" : cross_fscore,
        "id_precision" : id_precision,
        "id_recall" : id_recall,
        "id_fscore" : id_fscore,
        "gnv_map" : gnv_map,
        "cross_map" : cross_map,
        "id_map" : id_map, 
    })
    results_dict.update(meteor_dict)
    results_dict.update({"bleu": bleu_dict["bleu"]})
    return results_dict

def main():
    global tokenizer, rouge_metric, input_column, target_column, templater, meteor_metric, bleu_metric
    
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
        raise ValueError(f"The target column was not found in the train dataset, have: {args.target_column}, found: {dataset.column_names['train']}.")
    
    if has_valid_dataset and args.target_column not in dataset.column_names['valid']:
        raise ValueError(f"The target column was not found in the valid dataset, have: {args.target_column}, found: {dataset.column_names['valid']}.")
    
    if args.input_column not in dataset.column_names['train']:
        raise ValueError(f"The input column was not found in the train dataset, have: {args.input_column}, found: {dataset.column_names['train']}")
    
    if args.input_column not in dataset.column_names['valid']:
        raise ValueError(f"The input column was not found in the valid dataset, have: {args.input_column}, found: {dataset.column_names['valid']}")
    
    target_column = args.target_column
    input_column = args.input_column
    
    model_id = args.model
    template = BASE_MISTRAL_TEMPLATE
    if "llama" in model_id.lower():
        template = BASE_LLAMA_TEMPLATE
    
    templater = TemplateLLMQuerySender(None, template, start_seq='[', end_seq=']')
    
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
    bleu_metric = evaluate.load("bleu")
    meteor_metric = evaluate.load("meteor")
    
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