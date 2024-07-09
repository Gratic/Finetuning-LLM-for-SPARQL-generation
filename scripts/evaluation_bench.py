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
    execute_pipeline,
    extract_query,
    is_query_empty,
    add_relevant_prefixes_to_query,
    add_limit_clause,
    send_query_to_api,
)
import ast

def verify_file_exists(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The dataset file '{file_path}' does not exist.")

def load_and_verify_dataframe(file_path, generated_field, executed_field=None):
    df = load_dataset(file_path)

    if generated_field not in df.columns:
        raise ValueError(f"The column '{generated_field}' is missing from the DataFrame")

    if executed_field and executed_field not in df.columns:
        raise ValueError(f"The column '{executed_field}' is missing from the DataFrame")

    columns = [generated_field]
    if executed_field:
        columns.append(executed_field)

    df = df[columns]

    return df

def safe_eval(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            logging.warning(f"Could not evaluate string: {x}")
            return None
    return x

def execute_query(query: str, start_tag: str, end_tag: str):
    query = extract_query(query, start_tag, end_tag)
    if is_query_empty(query):
        return "exception: query is empty"
    
    query_with_prefixes = add_relevant_prefixes_to_query(query)
    query_with_limit = add_limit_clause(query_with_prefixes)
    
    response = send_query_to_api(query_with_limit, do_print=False)
        
    return response

def execute_all_queries(generated_texts, start_tag, end_tag):
    translated_preds = list(map(lambda x: x['output'], execute_pipeline(generated_texts)))
    return [execute_query(query, start_tag, end_tag) for query in translated_preds]

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

def process_executed_preds(executed_preds):
    exception_count = 0
    timeout_count = 0
    processed_preds = []

    for pred in executed_preds:
        if isinstance(pred, str):
            if pred.startswith('exception:'):
                exception_count += 1
                processed_preds.append(None)
            elif pred.startswith('timeout'):
                timeout_count += 1
                processed_preds.append(None)
            else:
                processed_preds.append(safe_eval(pred))
        else:
            processed_preds.append(pred)

    return processed_preds, exception_count, timeout_count

def main(args):
    nltk.download('wordnet', quiet=True)
    nltk.download("punkt", quiet=True)
    
    if args.execute_queries:
        to_evaluate = load_and_verify_dataframe(args.dataset, args.generated_field)
        generated_texts = to_evaluate[args.generated_field].to_list()
        executed_preds = execute_all_queries(generated_texts, args.start_tag, args.end_tag)
    else:
        to_evaluate = load_and_verify_dataframe(args.dataset, args.generated_field, args.executed_field)
        executed_preds = to_evaluate[args.executed_field].to_list()

    executed_preds, exception_count, timeout_count = process_executed_preds(executed_preds)
    
    generated_texts = to_evaluate[args.generated_field].to_list()
    
    gold_dataset = datasets.load_dataset(args.hf_dataset, split=args.hf_split)
    
    compute_metrics = create_compute_metrics(
        gold_dataset,
        target_column=args.hf_target,
        rouge_metric=evaluate.load("rouge"),
        bleu_metric=evaluate.load("bleu"),
        meteor_metric=evaluate.load("meteor")
        )
    
    results = compute_metrics(
        generated_texts=generated_texts,
        executed_preds=executed_preds
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
    parser.add_argument('-d', '--dataset', required=True, type=str, help="Path to the dataset file")
    parser.add_argument('-gq', '--generated-field', required=True, type=str, help="Column name for generated queries")
    parser.add_argument('-eq', '--executed-field', type=str, help="Column name for pre-executed query results")
    parser.add_argument('-hd', '--hf-dataset', required=True, type=str, help="HuggingFace dataset name")
    parser.add_argument('-hs', '--hf-split', required=True, type=str, help="HuggingFace dataset split")
    parser.add_argument('-ht', '--hf-target', required=True, type=str, help="Target field in HuggingFace dataset")
    parser.add_argument('-m', '--model', required=True, type=str, help="Model name for results labeling")
    parser.add_argument('-o', '--output', required=True, type=str, help="Output folder for results")
    parser.add_argument('-sn', '--save-name', required=True, type=str, help="Filename for saved results (without extension)")
    parser.add_argument("-log", "--log-level", type=str, default="warning", help="Logging level (debug, info, warning, error, critical)")
    parser.add_argument("-logf", "--log-file", type=str, default="", help="Log file path (if not specified, logs to console)")
    parser.add_argument("--execute-queries", action="store_true", help="Execute queries instead of using pre-executed results")
    parser.add_argument("--start-tag", type=str, default="<query>", help="Start tag for query extraction")
    parser.add_argument("--end-tag", type=str, default="</query>", help="End tag for query extraction")
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