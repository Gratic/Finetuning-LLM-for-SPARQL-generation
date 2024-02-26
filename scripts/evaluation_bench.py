import sys
from pathlib import Path
sys.path.append(Path("modules").absolute().__str__())

from nltk.translate.bleu_score import corpus_bleu
import argparse
import json
import logging
import nltk
import os
import pandas as pd
from evaluation_utils import compute_precision, compute_recall, corpus_meteor, average_precision
from data_utils import failed_generation_index, eval_dataset, get_nested_values, load_dataset, safe_loc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Evaluation bench for LLM",
                                    description="Evaluate LLMs for SPARQL generation")
    parser.add_argument('-d', '--dataset', required=True, type=str, help="The path to the dataset.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-g', '--gold', type=str, help="The path to the gold dataset (dataset with answers).")
    group.add_argument('-pg', '--preprocess-gold', type=str, help="The path to the preprocessed gold dataset (dataset with answers).")
    parser.add_argument('-m', '--model', required=True, type=str, help="The model name (used only to fill 'model_name' column of the results).")
    parser.add_argument('-o', '--output', required=True, type=str, help="Folder to output the results.")
    parser.add_argument('-sn', '--save-name', required=True, type=str, help="Name of the save file.")
    parser.add_argument("-log", "--log-level", type=str, help="Logging level (debug, info, warning, error, critical).", default="warning")
    parser.add_argument("-logf", "--log-file", type=str, help="Logging file.", default="")

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
    
    nltk.download('wordnet', quiet=True)
    
    df = load_dataset(args.dataset)
    df_no_gen_fail = df.drop(failed_generation_index(df))
    df_exec_timeout = df_no_gen_fail.loc[df_no_gen_fail['execution'] == 'timeout']
    df_exec_fail = df_no_gen_fail.loc[df_no_gen_fail['execution'].str.startswith('exception')]
    df_exec_empty = df_no_gen_fail.loc[df_no_gen_fail['execution'].isnull()]
    df_exec_to_eval = df_no_gen_fail.drop(df_exec_timeout.index).drop(df_exec_fail.index).drop(df_exec_empty.index)
    df_eval = eval_dataset(df_exec_to_eval)
    df_eval['get_nested_values'] = df_eval.apply(lambda x: get_nested_values(x['eval']), axis=1)
    
    df_gold_eval = None
    if args.gold != None:
        df_gold = load_dataset(args.gold)
        df_gold_exec_timeout = df_gold.loc[df_gold['execution'] == 'timeout']
        df_gold_exec_fail = df_gold.loc[df_gold['execution'].str.startswith('exception')]
        df_gold_exec_empty = df_gold.loc[df_gold['execution'].isnull()]
        df_gold_exec_to_eval = df_gold.drop(df_gold_exec_timeout.index).drop(df_gold_exec_fail.index).drop(df_gold_exec_empty.index)
        df_gold_eval = eval_dataset(df_gold_exec_to_eval, "gold_eval")
        df_gold_eval['gold_get_nested_values'] = df_gold_eval.apply(lambda x: get_nested_values(x['gold_eval']), axis=1)
    else:
        with open(args.preprocess_gold, "r") as f:
            data = json.load(f)
        df_gold_eval = pd.read_json(data['df_gold_eval'])
        
    
    df_merged_eval = df_eval.copy()
    
    # Merging manually
    df_merged_eval["gold_eval"] = df_merged_eval.apply(lambda x: safe_loc(x, df_gold_eval, "gold_eval", default=None), axis=1)
    df_merged_eval["gold_get_nested_values"] = df_merged_eval.apply(lambda x: safe_loc(x, df_gold_eval, "gold_get_nested_values", default=[]), axis=1)
    
    # Computing metrics
    df_merged_eval["precision"] = df_merged_eval.apply(lambda x: compute_precision(x['get_nested_values'], x['gold_get_nested_values']), axis=1)
    df_merged_eval["recall"] = df_merged_eval.apply(lambda x: compute_recall(x['get_nested_values'], x['gold_get_nested_values']), axis=1)
    df_merged_eval["average_precision"] = df_merged_eval.apply(lambda x: average_precision(x['get_nested_values'], x['gold_get_nested_values'], k_max=100000), axis=1)
    
    m_precision = df_merged_eval['precision'].mean()
    m_recall = df_merged_eval['recall'].mean()
    m_fscore = 2*m_precision*m_recall/(m_precision+m_recall)
    mean_average_precision = df_merged_eval['average_precision'].mean()

    bleu_score = corpus_bleu([[x.split()] for x in df_no_gen_fail['target_template']], [x.split() for x in df_no_gen_fail['translated_prompt']])
    meteor_score = corpus_meteor(df_no_gen_fail['target_template'], df_no_gen_fail['translated_prompt'])
    # TODO: add the correct syntax metric
    
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
                        "num_eval_empty": len(df_eval.loc[df_eval['eval'].map(len) == 0]),
                        "bleu_score": bleu_score,
                        "meteor_score": meteor_score,
                        "precision": m_precision,
                        "recall": m_recall,
                        "f1score": m_fscore,
                        "mean_average_precision": mean_average_precision,
                    })
    
    os.makedirs(args.output, exist_ok=True)
    serie.to_json(os.path.join(args.output, f"{args.save_name}.json"))