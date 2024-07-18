import sys
from pathlib import Path
sys.path.append(Path("modules").absolute().__str__())

from data_utils import get_nested_values, safe_eval, set_seed, make_dataframe_from_sparql_response
from datasets import load_dataset
from evaluation_utils import keep_id_columns, compute_metrics_for_two_list, compute_metrics_for_two_df
from execution_utils import is_query_empty, can_add_limit_clause, add_relevant_prefixes_to_query, send_query_to_api
from peft import LoraConfig
from prompts_template import PERSONA_BASIC_INSTRUCTION, BASE_MISTRAL_TEMPLATE, LLAMA2_TEMPLATE, BASE_BASIC_INSTRUCTION, ELABORATE_INSTRUCTION, CODELLAMA_TEMPLATE, get_template_for_model
from transformers import TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, DataCollatorForLanguageModeling, Seq2SeqTrainingArguments, GenerationConfig, DataCollatorForSeq2Seq, EvalPrediction
from transformers.training_args import OptimizerNames, SchedulerType
from huggingface_hub import login
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from modified_trainer import SFTTrainerGen, CustomSFTConfig
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
import bitsandbytes as bnb
from copy import deepcopy
import random
from accelerate import Accelerator
import json

input_column = None
target_column = None
templater: TemplateLLMQuerySender = None
start_tag = None
end_tag = None
# TODO: delete counter below
counter = 0

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
    if not query.startswith(('PREFIX', 'SELECT', 'prefix', 'select')):
        return False
    return True

def format_prompt_packing(example):
    text = generate_instruction_prompt(
            prompt=example[input_column],
            target=example[target_column]
            )
    return text

# def format_prompt(example):
#     output_texts = []
#     for i in range(len(example[input_column])):
#         text = generate_instruction_prompt(
#             prompt=example[input_column][i][0],
#             target=example[target_column][i],
#             system_prompt=ELABORATE_INSTRUCTION,
#             )
#         output_texts.append(text)

#     return output_texts

def create_format_prompt(input_col:str, target_col:str, with_target: bool, template:str, start_tag:str, end_tag:str, tokenizer:AutoTokenizer):
    def format_prompt(examples):
        output_texts = []
        label_texts = []
        for i in range(len(examples[input_col])):
            n = len(examples[input_col][i])
            
            text = template.replace('[system_prompt]', ELABORATE_INSTRUCTION)
            text = text.replace('[prompt]', examples[input_col][i][random.randint(0, n-1)])
            
            # TODO: can try to lead answer
            answer_text = f'\n{start_tag}{examples[target_col][i]}{end_tag}'
            
            if with_target:
                text += answer_text
        
            output_texts.append(text)
            label_texts.append(answer_text)
            
        if with_target:        
            return {"text": output_texts}
        else:
            # This part is different because Trainer doesn't generate labels for eval ?
            # Maybe due to DataCollator
            examples["input_ids"] = tokenizer(output_texts,
                                             padding=False,
                                             truncation=False,
                                             add_special_tokens=False)["input_ids"]
            examples["labels"] = tokenizer(label_texts,
                                             padding=False,
                                             truncation=False,
                                             add_special_tokens=False)["input_ids"]
            return examples
    return format_prompt

# LEGACY CODE vvv
def generate_instruction_prompt(prompt: str, target: str, system_prompt:str = BASE_BASIC_INSTRUCTION):
    text = templater.apply_template({
            "system_prompt": system_prompt,
            "prompt": prompt,
        })
    text += f"{start_tag}{target}{end_tag}"
    return text

def parse_args(list_args=None):
    parser = argparse.ArgumentParser(prog="PEFT (QLora) SFT Script")
    parser.add_argument("-m", "--model", type=str, help="Huggingface model or path to a model to finetune.", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("-op", "--optimizer", type=str, help="Huggingface implemented optimizers. Example: adamw_torch, adamw_bnb_8bit.", default="adamw_torch")
    parser.add_argument("-lr", "--learning-rate", type=float, help="Learning rate.", default=1e-5)
    parser.add_argument("-ctx", "--context-length", type=int, help="Maximum context length.", default=2048)
    parser.add_argument("-mq", "--model-quant", type=str, help="How should the model be quantized. Choices available are 'no', '4bit' and '8bit'.", default='no', choices=['no', '4bit', '8bit'])
    parser.add_argument("-cd", "--computational-datatype", type=str, help="Define the computational datatype used for training: fp16, bf16, fp32.", default='no', choices=['fp16', 'bf16', 'fp32'])
    parser.add_argument("-trg", "--target-column", type=str, help="Indicates which column to use for answers (default= 'target_template').", default="target_template")
    parser.add_argument("-ic", "--input-column", type=str, help="Indicates which column to use for the input prompt (default= 'basic_input').", default="basic_input")
    parser.add_argument("-hfd", "--hf-dataset", type=str, help="Path to the huggingface dataset.", default="")
    parser.add_argument("-st", "--start-tag", type=str, help="Prefix the answer.", default="[query]")
    parser.add_argument("-et", "--end-tag", type=str, help="Suffix the answer.", default="[/query]")
    parser.add_argument("-np", "--no-peft", action="store_true", help="Disable LoRA adapters.")
    parser.add_argument("-rv", "--rvalue", type=int, help="Lora r-value.", default=8)
    parser.add_argument("-ra", "--ralpha", type=float, help="Lora r-alpha multiplier based on r-value. r-alpha = r-value * multiplier (this)", default=1.)
    parser.add_argument("-ld", "--lora-dropout", type=float, help="Lora dropout value.", default=0.05)
    parser.add_argument("-bs", "--batch-size", type=int, help="Batch size for training.", default=1)
    parser.add_argument("-ga", "--gradient-accumulation", type=int, help="Gradient accumulation, number of batch to process before making an optimizer step.", default=4)
    parser.add_argument("-gc", "--gradient-checkpointing", type=int, help="Turn on gradient checkpointing (1=True, 0=False).", default=1)
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
    parser.add_argument("-ne", "--no-eval", action="store_true", help="Don't do evaluation.")
    parser.add_argument("-tok", "--token", type=str, help="Auth token for gated models (like LLaMa 2).", default="")
    
    args = parser.parse_args(list_args)
    
    args.start_tag = args.start_tag.replace("\\n", "\n")
    args.end_tag = args.end_tag.replace("\\n", "\n")
    return args

def extract_query(query: str, start_tag: str, end_tag: str):
    start_sparql = query.find(start_tag)
    end_sparql = query.rfind(end_tag, start_sparql+len(start_tag))
    
    if start_sparql != -1 and end_sparql != -1:
        return query[start_sparql+len(start_tag):end_sparql]
    return None

def add_limit_clause(query: str) -> str:
    if can_add_limit_clause(query):
        return f"{query}\nLIMIT 10"
    return query

def execute_query(query: str, start_tag: str, end_tag: str):
    query = extract_query(query, start_tag, end_tag)
    if is_query_empty(query):
        return None
    
    query_with_prefixes = add_relevant_prefixes_to_query(query)
    query_with_limit = add_limit_clause(query_with_prefixes)
    
    response = send_query_to_api(query_with_limit, do_print=False)
    
    if isinstance(response, str) and response.startswith(('exception:', 'timeout')):
        return None
        
    return safe_eval(response) if isinstance(response, str) else response

def create_empty_results():
    metrics = ["rouge1", "rouge2", "rougeL", "rougeLsum", "correct_syntax", 
               "gnv_precision", "gnv_recall", "gnv_rr", "cross_precision", "cross_recall", 
               "cross_rr", "id_precision", "id_recall", "id_rr", "gnv_map", "cross_map", 
               "id_map", "bleu", "meteor", "gnv_overlap", "gnv_jaccard", "gnv_dice_coeff", 
               "cross_overlap", "cross_jaccard", "cross_dice_coeff", "id_overlap", 
               "id_jaccard", "id_dice_coeff"]
    return {metric: 0. for metric in metrics}

def align_data_lengths(generated_texts, target_queries, raw_target_queries, executed_target_queries):
    min_length = min(len(generated_texts), len(raw_target_queries))
    return (
        generated_texts[:min_length],
        target_queries[:min_length],
        raw_target_queries[:min_length],
        executed_target_queries[:min_length]
    )
    
def tokenize_predictions(generated_texts):
    return ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in generated_texts]

def save_debug_data(preds, generated_texts, tokenized_preds, target_queries, raw_target_queries, str_labels, str_inputs):
    if 'args' in globals():
        save_data = {
            "preds": preds.tolist(),
            "generated_texts": generated_texts,
            "tokenized_preds": tokenized_preds,
            "target_queries": target_queries,
            "raw_target_queries": raw_target_queries,
            "str_labels": str_labels,
            "str_inputs": str_inputs,
        }
        save_path = Path(os.path.join(args.output, f"{args.save_name}_compute_metrics_{globals().get('counter', 0)}.json"))
        save_path.write_text(json.dumps(save_data))
        globals()['counter'] = globals().get('counter', 0) + 1

def process_predictions(eval_preds, tokenizer):
    preds, labels, inputs = eval_preds.predictions, eval_preds.label_ids, eval_preds.inputs
    pad_token_id = tokenizer.pad_token_id
    preds = np.where(preds != -100, preds, pad_token_id)
    labels = np.where(labels != -100, labels, pad_token_id)
    inputs = np.where(inputs != -100, inputs, pad_token_id)
    return preds, labels, inputs

def decode_texts(preds, labels, inputs, tokenizer):
    generated_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
    str_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    str_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
    return generated_texts, str_labels, str_inputs

def get_target_data(eval_dataset, target_column):
    return (
        list(eval_dataset[target_column]),
        list(eval_dataset['target_raw']),
        list(eval_dataset['gold_execution'])
    )

def create_and_process_dataframe(executed_target_queries, executed_preds):
    data = pd.DataFrame({
        "executed_labels": map(lambda x: eval(x), executed_target_queries),
        "executed_preds": executed_preds,
    })
    
    data['get_nested_values_labels'] = data.apply(lambda x: get_nested_values(x["executed_labels"]), axis=1)
    data['get_nested_values_preds'] = data.apply(lambda x: get_nested_values(x["executed_preds"]), axis=1)
    
    df_not_null = pd.DataFrame()
    df_not_null['labels_df'] = data.apply(lambda x: make_dataframe_from_sparql_response(x['executed_labels']) if isinstance(x['executed_labels'], list) else pd.DataFrame(), axis=1)
    df_not_null['preds_df'] = data.apply(lambda x: make_dataframe_from_sparql_response(x['executed_preds']) if isinstance(x['executed_preds'], list) else pd.DataFrame(), axis=1)
    
    df_not_null['labels_id_columns'] = df_not_null.apply(lambda x: keep_id_columns(x['labels_df']), axis=1)
    df_not_null['preds_id_columns'] = df_not_null.apply(lambda x: keep_id_columns(x['preds_df']), axis=1)
    
    return data, df_not_null

def compute_all_metrics(data, df_not_null):
    data['nested_metrics'] = data.apply(lambda x: compute_metrics_for_two_list(results=x['get_nested_values_preds'], gold=x['get_nested_values_labels'], k=5), axis=1)
    df_not_null['cross_metrics'] = df_not_null.apply(lambda x: compute_metrics_for_two_df(results=x['preds_df'], gold=x['labels_df'], k=5), axis=1)
    df_not_null['id_metrics'] = df_not_null.apply(lambda x: compute_metrics_for_two_df(results=x['preds_id_columns'], gold=x['labels_id_columns'], k=5), axis=1)
    
    nested_metrics = pd.DataFrame(data['nested_metrics'].map(lambda x: x._asdict()).to_list())
    cross_metrics = pd.DataFrame(df_not_null['cross_metrics'].map(lambda x: x._asdict()).to_list())
    id_metrics = pd.DataFrame(df_not_null['id_metrics'].map(lambda x: x._asdict()).to_list())
    
    return nested_metrics, cross_metrics, id_metrics

def calculate_final_metrics(nested_metrics, cross_metrics, id_metrics, tokenized_preds, target_queries, executed_preds, rouge_metric, bleu_metric, meteor_metric):
    results_dict = {}
    
    for prefix, metrics in [("gnv", nested_metrics), ("cross", cross_metrics), ("id", id_metrics)]:
        for metric in ['mean_average_precision', 'precision_k', 'recall_k', 'mean_reciprocal_rank', 'overlap', 'jaccard', 'dice_coeff']:
            results_dict[f"{prefix}_{metric.replace('_k', '').replace('mean_average_precision', 'map').replace('mean_reciprocal_rank', 'rr')}"] = metrics[metric].mean()

    results_dict.update(rouge_metric.compute(predictions=tokenized_preds, references=target_queries, use_stemmer=True))
    results_dict.update(meteor_metric.compute(predictions=tokenized_preds, references=target_queries))
    results_dict["bleu"] = bleu_metric.compute(predictions=tokenized_preds, references=target_queries)["bleu"]
    results_dict["correct_syntax"] = sum(x is not None for x in executed_preds) / len(executed_preds)

    return results_dict

def create_compute_metrics(eval_dataset, tokenizer, target_column, rouge_metric, bleu_metric, meteor_metric, start_tag, end_tag):
    # https://huggingface.co/docs/evaluate/transformers_integrations
    
    def compute_metrics(eval_preds: EvalPrediction):
        preds, labels, inputs = process_predictions(eval_preds, tokenizer)
        generated_texts, str_labels, str_inputs = decode_texts(preds, labels, inputs, tokenizer)
        
        # All generated_texts are empty.
        if all(len(x) == 0 for x in generated_texts):
            return create_empty_results()
        
        target_queries, raw_target_queries, executed_target_queries = get_target_data(eval_dataset, target_column)
        
        generated_texts, target_queries, raw_target_queries, executed_target_queries = align_data_lengths(
            generated_texts, target_queries, raw_target_queries, executed_target_queries
        )
        
        tokenized_preds = tokenize_predictions(generated_texts)
        
        save_debug_data(preds, generated_texts, tokenized_preds, target_queries, raw_target_queries, str_labels, str_inputs)
            
        translated_preds = list(map(lambda x: x['output'], execute_pipeline(generated_texts)))
        executed_preds = [execute_query(query, start_tag, end_tag) for query in translated_preds]
        
        data, df_not_null = create_and_process_dataframe(executed_target_queries, executed_preds)
        
        nested_metrics, cross_metrics, id_metrics = compute_all_metrics(data, df_not_null)
        
        results_dict = calculate_final_metrics(
            nested_metrics, cross_metrics, id_metrics, 
            tokenized_preds, target_queries, executed_preds,
            rouge_metric, bleu_metric, meteor_metric
        )
        
        return results_dict
    return compute_metrics

def main(args):
    global input_column, target_column, templater, start_tag, end_tag
        
    setup_logging(args)
    
    if args.random_seed != 0:
        set_seed(args.random_seed)
    
    accelerator = Accelerator()
    
    has_valid_dataset = False
    with accelerator.main_process_first():
        dataset = load_dataset(args.hf_dataset, token=args.token)
        has_valid_dataset = any([x in dataset.column_names.keys() for x in ["valid"]])
    
    save_path_adapters = os.path.join(args.output, f"{args.save_name}_adapters")
    
    do_packing = bool(args.packing)
    
    if args.target_column not in dataset.column_names['train']:
        raise ValueError(f"The target column was not found in the train dataset, have: {args.target_column}, found: {dataset.column_names['train']}.")
    
    if args.input_column not in dataset.column_names['train']:
        raise ValueError(f"The input column was not found in the train dataset, have: {args.input_column}, found: {dataset.column_names['train']}")
    
    if args.context_length <= 0:
        raise ValueError(f"The context length must be strictly positive, found: {args.context_length}")
    
    target_column = args.target_column
    input_column = args.input_column
    start_tag = args.start_tag
    end_tag = args.end_tag
    
    model_id = args.model
    template = get_template_for_model(model_id)
        
    if args.token != "":
        login(token=args.token)
        
    nltk.download("punkt", quiet=True)
    
    templater = TemplateLLMQuerySender(None, template, start_seq='[', end_seq=']')
    
    os.environ["WANDB_PROJECT"] = args.wnb_project  # name your W&B project
    os.environ["WANDB_LOG_MODEL"] = args.wnb_log  # log all model checkpoints
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    args_computational_datatype_to_pytorch_dict = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16
    }
    
    # https://huggingface.co/docs/transformers/main_classes/quantization#transformers.BitsAndBytesConfig
    bnb_config = None
    if args.model_quant == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=args_computational_datatype_to_pytorch_dict[args.computational_datatype],
        )
    elif args.model_quant == "8bit":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_has_fp16_weight=False,
        )
    
    logging.info(f"Loading model: {model_id}.")
    print(f"Loading model: {model_id}.")
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map = {"": Accelerator().local_process_index}
        # device_map="auto", # torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    
    logging.info(f"Loading tokenizer: {model_id}.")
    print(f"Loading tokenizer: {model_id}.")
    # https://medium.com/@parikshitsaikia1619/mistral-mastery-fine-tuning-fast-inference-guide-62e163198b06
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    pretrained_model.resize_token_embeddings(len(tokenizer))
    pretrained_model.config.pad_token_id = tokenizer.pad_token_id
    
    response_template="[/INST]"
    if "llama-3" in model_id.lower():
        response_template="<|start_header_id|>assistant<|end_header_id|>"
    if "codellama" in model_id.lower():
        response_template=[29961, 29914, 25580, 29962] # seems to depend on context so had to find the right ones after tokenized it.
    
    left_sided_tokenizer = None
    if tokenizer.padding_side == "right":
        left_sided_tokenizer = deepcopy(tokenizer)
        left_sided_tokenizer.padding_side = "left"
        # eval_collator = DataCollatorForLanguageModeling(tokenizer=left_sided_tokenizer, mlm=False)
        eval_collator = DataCollatorForSeq2Seq(tokenizer=left_sided_tokenizer)
    else:
        # eval_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        eval_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    
    # TODO: normal collator, without prompt masking
    # TODO: make it a parameter
    collator_args = {
        "response_template": response_template,
        "mlm": False,
    }
    collator = None if do_packing else DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, **collator_args)
    # collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    # stop_strings = ["<|endoftext|>", "<|end|>", "</s>", "<|eot_id|>", "<|end_of_text|>"]
    training_args = CustomSFTConfig(
        bf16=args.computational_datatype == 'bf16', # Computational dtype of the weights of the adapter
        bf16_full_eval=args.computational_datatype == 'bf16',
        fp16=args.computational_datatype == 'fp16',
        fp16_full_eval=args.computational_datatype == 'fp16',
        output_dir=save_path_adapters,
        # optim=args.optimizer,
        optim=OptimizerNames.PAGED_ADAMW_8BIT,
        lr_scheduler_type=SchedulerType.COSINE,
        warmup_ratio=0.1,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else {},
        neftune_noise_alpha=args.neft_tune_alpha if args.neft_tune_alpha != 0 else None,
        max_seq_length=args.context_length,
        dataset_text_field="text",
        eval_on_start=False,
        # generation_config=GenerationConfig(do_sample=True,
        #                                    top_p=0.95,
        #                                    temperature=0.2,
        #                                    max_new_tokens=512,
        #                                    num_beams=1,
                                            #  use_cache=True,
                                            #  stop_strings=stop_strings,
                                            #  eos_token_id=tokenizer.eos_token_id,
                                            #  pad_token_id=tokenizer.pad_token_id,
        #                                    ),
        generation_config=GenerationConfig(do_sample=False,
                                           max_new_tokens=512,
                                           num_beams=1,
                                           use_cache=True,
                                        #    stop_strings=stop_strings,
                                           eos_token_id=tokenizer.eos_token_id,
                                           pad_token_id=tokenizer.pad_token_id,
                                           ),
        # generation_max_length=512,
        generation_num_beams=1,
        predict_with_generate=True,
        include_inputs_for_metrics=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
        dataloader_drop_last=False,
        dataset_num_proc=1,
        eval_strategy="epoch" if has_valid_dataset and not args.no_eval else "no",
        num_train_epochs=args.epochs,
        packing=do_packing,
        save_strategy="no", # TODO: maybe save checkpoints and do evaluation with them later
        logging_strategy="steps",
        logging_steps=10,
        run_name=args.run_name,
        report_to="wandb",
        seed=args.random_seed
    )    

    lora_config = LoraConfig(
        r=args.rvalue,
        lora_alpha=args.rvalue*args.ralpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
    ) if not args.no_peft else None

    
    
    # optimizer = bnb.optim.Adam8bit(pretrained_model.parameters(), lr=1e-5)
    # optimizer = torch.optim.AdamW(pretrained_model.parameters(), lr=1e-5)
    with accelerator.main_process_first():
        train_formatting_func = create_format_prompt(
            input_col=input_column,
            target_col=target_column,
            with_target=True,
            template=get_template_for_model(model_id),
            start_tag=args.start_tag,
            end_tag=args.end_tag,
            tokenizer=tokenizer,
        )
        
        eval_formatting_func = create_format_prompt(
            input_col=input_column,
            target_col=target_column,
            with_target=False,
            template=get_template_for_model(model_id),
            start_tag=args.start_tag,
            end_tag=args.end_tag,
            tokenizer=left_sided_tokenizer,
        )
        
        train_dataset = dataset["train"]
        valid_dataset = dataset["valid"] if has_valid_dataset else None
        
        train_dataset = train_dataset.map(train_formatting_func, batched=True, load_from_cache_file=False)
        valid_dataset = valid_dataset.map(eval_formatting_func, batched=True, load_from_cache_file=False) if valid_dataset else None
    
    compute_metrics = create_compute_metrics(
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        target_column=target_column,
        rouge_metric=evaluate.load("rouge"),
        bleu_metric=evaluate.load("bleu"),
        meteor_metric=evaluate.load("meteor"),
        start_tag=start_tag,
        end_tag=end_tag,
        )
    
    trainer = SFTTrainerGen(
        pretrained_model,
        args=training_args,
        tokenizer=tokenizer,
        # optimizers=(optimizer, None),
        data_collator=collator,
        eval_data_collator=eval_collator,
        left_sided_tokenizer=left_sided_tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        # formatting_func= format_prompt_packing if do_packing else train_formatting_func,
        peft_config=lora_config,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
    )
    
    print_trainable_parameters(trainer.model)

    logging.info(f"Starting training.")
    print("Starting training.")
    trainer.train()
        
    if args.save_adapters:
        # remove padding token
        pretrained_model.resize_token_embeddings(len(tokenizer) - 1)
        pretrained_model.config.pad_token_id = None
        
        logging.info(f"Saving adapters.")
        print("Saving adapters.")
        trainer.model.save_pretrained(save_path_adapters)

def setup_logging(args):
    numeric_log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_log_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}.")
    # TODO: rework logging
    # logging.basicConfig(filename=args.log_file if args.log_file else None, level=numeric_log_level)

if __name__ == "__main__":
    global args
    args = parse_args()
    main(args)