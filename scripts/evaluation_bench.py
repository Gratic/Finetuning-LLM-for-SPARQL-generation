import sys
from pathlib import Path
sys.path.append(Path("modules").absolute().__str__())

from evaluation_utils import (
    is_correct_SPARQL_query,
    load_and_merge_evaluation_and_gold_dataset,
    compute_metrics_for_two_df,
    compute_metrics_for_two_list,
)
import argparse
import evaluate
import logging
import nltk
import os
import pandas as pd
from data_utils import load_dataset
import datasets
from typing import List, Any
from sft_peft import (
    create_empty_results,
    get_target_data,
    align_data_lengths,
    tokenize_predictions,
    create_and_process_dataframe,
    compute_all_metrics,
    calculate_final_metrics,
)
import ast

def verify_file_exists(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The dataset file '{file_path}' does not exist.")

def load_and_verify_dataframe(file_path, generated_field, executed_field):
    df = load_dataset(file_path)

    missing_columns = []
    if generated_field not in df.columns:
        missing_columns.append(generated_field)
    if executed_field not in df.columns:
        missing_columns.append(executed_field)

    if missing_columns:
        raise ValueError(f"The following columns are missing from the DataFrame: {', '.join(missing_columns)}")

    df = df[[generated_field, executed_field]]

    return df

def safe_eval(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            logging.warning(f"Could not evaluate string: {x}")
            return None
    return x

def create_compute_metrics(eval_dataset, target_column, rouge_metric, bleu_metric, meteor_metric):
    # https://huggingface.co/docs/evaluate/transformers_integrations
    
    def compute_metrics(generated_texts: List[str], executed_preds: List[Any]):
        # All generated_texts are empty.
        if all(len(x) == 0 for x in generated_texts):
            return create_empty_results()
        
        target_queries, raw_target_queries, executed_target_queries = get_target_data(eval_dataset, target_column)
        
        generated_texts, target_queries, raw_target_queries, executed_target_queries = align_data_lengths(
            generated_texts, target_queries, raw_target_queries, executed_target_queries
        )
        
        tokenized_preds = tokenize_predictions(generated_texts)
        
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
    nltk.download('wordnet', quiet=True)
    nltk.download("punkt", quiet=True)
    
    executed_field = args.executed_field
    to_evaluate = load_and_verify_dataframe(args.dataset, args.generated_field, executed_field)
    gold_dataset = datasets.load_dataset(args.hf_dataset, split=args.hf_split)
    
    exception_count = to_evaluate[executed_field].str.startswith('exception:').sum()
    timeout_count = to_evaluate[executed_field].str.startswith('timeout').sum()

    to_evaluate.loc[to_evaluate[executed_field].str.startswith('exception:'), executed_field] = None
    to_evaluate.loc[to_evaluate[executed_field].str.startswith('timeout'), executed_field] = None
    
    to_evaluate[executed_field] = to_evaluate[executed_field].apply(safe_eval)
    
    compute_metrics = create_compute_metrics(
        gold_dataset,
        target_column=args.hf_target,
        rouge_metric=evaluate.load("rouge"),
        bleu_metric=evaluate.load("bleu"),
        meteor_metric=evaluate.load("meteor")
        )
    
    results = compute_metrics(
        generated_texts=to_evaluate[args.generated_field].to_list(),
        executed_preds=to_evaluate[executed_field].to_list()
    )
    
    results['model_name'] = args.model
    results['timeout_count'] = timeout_count
    results['exception_count'] = exception_count
    
    results_series = pd.Series(results)
    
    os.makedirs(args.output, exist_ok=True)
    
    output_path = os.path.join(args.output, f"{args.save_name}.json")
    results_series.to_json(output_path)
    
    print(f"Results saved to {output_path}")
    

def create_parser():
    parser = argparse.ArgumentParser(prog="Evaluation bench for LLM",
                                    description="Evaluate LLMs for SPARQL generation")
    parser.add_argument('-d', '--dataset', required=True, type=str, help="The path to the generated and executed query")
    parser.add_argument('-gq', '--generated-field', required=True, type=str, help="The name of the column in dataset where the generated queries are.")
    parser.add_argument('-eq', '--executed-field', required=True, type=str, help="The name of the column in dataset where the execution results of the queries are.")
    parser.add_argument('-hd', '--hf-dataset', required=True, type=str, help="The path to the dataset.")
    parser.add_argument('-hs', '--hf-split', required=True, type=str, help="The split of the huggingface dataset.")
    parser.add_argument('-ht', '--hf-target', required=True, type=str, help="The target field.")
    parser.add_argument('-m', '--model', required=True, type=str, help="The model name (used only to fill 'model_name' column of the results).")
    parser.add_argument('-o', '--output', required=True, type=str, help="Folder to output the results.")
    parser.add_argument('-sn', '--save-name', required=True, type=str, help="Name of the save file.")
    parser.add_argument("-log", "--log-level", type=str, help="Logging level (debug, info, warning, error, critical).", default="warning")
    parser.add_argument("-logf", "--log-file", type=str, help="Logging file.", default="")
    return parser

if __name__ == "__main__":
    parser = create_parser()

    args = parser.parse_args()
        
    numeric_log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_log_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}.")
    logging.basicConfig(filename=args.log_file if args.log_file else None, level=numeric_log_level)
    
    verify_file_exists(args.dataset)
    
    main(args)