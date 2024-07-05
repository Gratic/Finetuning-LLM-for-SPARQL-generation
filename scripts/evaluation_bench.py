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

def main(args):
    nltk.download('wordnet', quiet=True)
    nltk.download("punkt", quiet=True)
    rouge_metric = evaluate.load("rouge")
    bleu_metric = evaluate.load("bleu")
    meteor_metric = evaluate.load("meteor")
    
    df, df_exec_timeout, df_exec_fail, df_exec_empty, df_exec_to_eval, df_eval, df_gold_eval, df_gold_exec_timeout, df_gold_exec_fail, df_gold_exec_empty, df_gold_exec_to_eval, df_merged_eval = load_and_merge_evaluation_and_gold_dataset(args)
    
    if not df_merged_eval.empty:
        # Computing metrics using scikit-learn
        df_merged_eval['nested_metrics'] = df_merged_eval.apply(lambda x: compute_metrics_for_two_list(results=x['get_nested_values'], gold=x['gold_get_nested_values'], k=5), axis=1)
        df_merged_eval['cross_metrics'] = df_merged_eval.apply(lambda x: compute_metrics_for_two_df(results=x['eval_df'], gold=x['gold_eval_df'], k=5), axis=1)
        df_merged_eval['id_metrics'] = df_merged_eval.apply(lambda x: compute_metrics_for_two_df(results=x['id_columns'], gold=x['gold_id_columns'], k=5), axis=1)
        
        nested_metrics = pd.DataFrame(data=df_merged_eval['nested_metrics'].map(lambda x: x._asdict()).to_list())
        cross_metrics = pd.DataFrame(data=df_merged_eval['cross_metrics'].map(lambda x: x._asdict()).to_list())
        id_metrics = pd.DataFrame(data=df_merged_eval['id_metrics'].map(lambda x: x._asdict()).to_list())
        
        gnv_map = nested_metrics['mean_average_precision'].mean()
        gnv_precision = nested_metrics['precision_k'].mean()
        gnv_recall = nested_metrics['recall_k'].mean()
        gnv_rr = nested_metrics['mean_reciprocal_rank'].mean()
        gnv_overlap = nested_metrics['overlap'].mean()
        gnv_jaccard = nested_metrics['jaccard'].mean()
        gnv_dice_coeff = nested_metrics['dice_coeff'].mean()

        cross_map = cross_metrics['mean_average_precision'].mean()
        cross_precision = cross_metrics['precision_k'].mean()
        cross_recall = cross_metrics['recall_k'].mean()
        cross_rr = cross_metrics['mean_reciprocal_rank'].mean()
        cross_overlap = cross_metrics['overlap'].mean()
        cross_jaccard = cross_metrics['jaccard'].mean()
        cross_dice_coeff = cross_metrics['dice_coeff'].mean()

        id_map = id_metrics['mean_average_precision'].mean()
        id_precision = id_metrics['precision_k'].mean()
        id_recall = id_metrics['recall_k'].mean()
        id_rr = id_metrics['mean_reciprocal_rank'].mean()
        id_overlap = id_metrics['overlap'].mean()
        id_jaccard = id_metrics['jaccard'].mean()
        id_dice_coeff = id_metrics['dice_coeff'].mean()

        decoded_labels = df_merged_eval['target_raw'].map(lambda x: "\n".join(nltk.sent_tokenize(x.strip()))).to_list()
        decoded_preds = df_merged_eval['output'].map(lambda x: "\n".join(nltk.sent_tokenize(x.strip()))).to_list()

        rouge_dict = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # bleu_score = corpus_bleu([[x.split()] for x in df['target_template']], [x.split() for x in df['translated_prompt']])
        bleu_dict = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        # meteor_dict = corpus_meteor(hypotheses=decoded_preds, references=decoded_labels)
        meteor_dict = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)
        correct_syntax = len(df_exec_fail) / len(df)
    else:
        bleu_dict = {"bleu": 0.0}
        meteor_dict = {"meteor": 0.0}
        rouge_dict = {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeLsum": 0.0,
            "rougeL": 0.0
        }
        gnv_precision = 0.0
        gnv_recall = 0.0
        gnv_rr = 0.0
        gnv_map = 0.0
        id_precision = 0.0
        id_recall = 0.0
        id_rr = 0.0
        id_map = 0.0
        cross_precision = 0.0
        cross_recall = 0.0
        cross_rr = 0.0
        cross_map = 0.0
        correct_syntax = 0.0
        gnv_overlap = 0.0
        gnv_jaccard = 0.0
        gnv_dice_coeff = 0.0
        cross_overlap = 0.0
        cross_jaccard = 0.0
        cross_dice_coeff = 0.0
        id_overlap = 0.0
        id_jaccard = 0.0
        id_dice_coeff = 0.0
        
    serie = pd.Series(data=
    {
        "model_name": args.model,
        "num_rows": len(df),
        "num_gen_fail": len(df.loc[df['has_error'] == True]),
        "num_exec_timeout": len(df_exec_timeout),
        "num_exec_fail": len(df_exec_fail),
        "num_exec_empty": len(df_exec_empty),
        "num_exec_to_eval": len(df_exec_to_eval),
        "num_eval": len(df_eval),
        "num_eval_empty": len(df_eval.loc[df_eval['eval'].map(len) == 0]) if not df_eval.empty else 0,
        "gold_num_rows": len(df_gold_eval),
        "gold_num_exec_timeout": len(df_gold_exec_timeout),
        "gold_num_exec_fail": len(df_gold_exec_fail),
        "gold_num_exec_empty": len(df_gold_exec_empty),
        "gold_num_exec_to_eval": len(df_gold_exec_to_eval),
        "gold_num_eval_empty": len(df_gold_eval.loc[df_gold_eval['gold_eval'].map(len) == 0]) if not df_gold_eval.empty else 0,
        "bleu_score": bleu_dict["bleu"],
        "meteor_score": meteor_dict['meteor'],
        **rouge_dict,
        "get_nested_values_precision": gnv_precision,
        "get_nested_values_recall": gnv_recall,
        "get_nested_values_mean_reciprocal_rank": gnv_rr,
        "get_nested_values_mean_average_precision": gnv_map,
        "id_precision": id_precision,
        "id_recall": id_recall,
        "id_mean_reciprocal_rank": id_rr,
        "id_mean_average_precision": id_map,
        "cross_precision": cross_precision,
        "cross_recall": cross_recall,
        "cross_mean_reciprocal_rank": cross_rr,
        "cross_mean_average_precision": cross_map,
        "correct_syntax": correct_syntax,
        "gnv_overlap": gnv_overlap,
        "gnv_jaccard": gnv_jaccard,
        "gnv_dice_coeff": gnv_dice_coeff,
        "cross_overlap": cross_overlap,
        "cross_jaccard": cross_jaccard,
        "cross_dice_coeff": cross_dice_coeff,
        "id_overlap": id_overlap,
        "id_jaccard": id_jaccard,
        "id_dice_coeff": id_dice_coeff,
    })
    
    os.makedirs(args.output, exist_ok=True)
    serie.to_json(os.path.join(args.output, f"{args.save_name}.json"))

def create_parser():
    parser = argparse.ArgumentParser(prog="Evaluation bench for LLM",
                                    description="Evaluate LLMs for SPARQL generation")
    parser.add_argument('-d', '--dataset', required=True, type=str, help="The path to the dataset.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-g', '--gold', type=str, help="The path to the gold dataset (dataset with answers).")
    group.add_argument('-ghf', '--hf-gold', type=str, help="The path to the gold dataset huggingface.")
    group.add_argument('-pg', '--preprocess-gold', type=str, help="The path to the preprocessed gold dataset (dataset with answers).")
    
    parser.add_argument('-ghfs', '--hf-gold-split', type=str, help="The split of the huggingface dataset.")
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

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"The dataset file not found with path: {args.dataset}")

    if args.gold != None and not os.path.exists(args.gold):
        raise FileNotFoundError(f"The gold dataset file not found with path: {args.gold}")
    
    if args.preprocess_gold != None and not os.path.exists(args.preprocess_gold):
        raise FileNotFoundError(f"The preprocess gold dataset file not found with path: {args.preprocess_gold}")
    
    main(args)