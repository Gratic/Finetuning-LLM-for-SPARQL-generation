from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import single_meteor_score
from typing import List, Dict, Union
import argparse
import logging
import os
import pandas as pd
from ast import literal_eval

def failed_generation_index(dataset: pd.DataFrame):
    return dataset.loc[dataset['has_error'] == True].index

def corpus_meteor(references: List, hypotheses: List):
    meteor_scores = 0.
    for ref, hyp in zip(references, hypotheses):
        meteor_scores += single_meteor_score(ref, hyp)
    return meteor_scores / float(len(references))

def safe_eval(execution: str):
    """Evaluates """
    try:
        return literal_eval(execution)
    except Exception as inst:
        logging.error(f"Exception occured while evaluating: {inst}.")
        print(f"Exception occured while evaluating: {inst}.")
        return None

def eval_dataset(dataset: pd.DataFrame, col_name: str = "eval"):
    df_eval = dataset.copy()
    df_eval[col_name] = df_eval.apply(lambda x: safe_eval(x['execution']), axis=1)
    return df_eval[~df_eval[col_name].isnull()]

def get_nested_values(element: Union[Dict, str]):
    values = []
    if isinstance(element, dict):
        for k, v in element.items():
            if isinstance(v, dict):
                values += get_nested_values(v)
            elif isinstance(v, str):
                if 'value' in k:
                    values.append(v)
    elif isinstance(element, list):
        for el in element:
            values += get_nested_values(el)
    else:
        logging.error(f"get_nested_values doesn't have an implementation for: {type(element)}.")
        raise TypeError(f"Compatible types are Dict and List, found: {type(element)}.")
    return values

def compute_precision(hypothesis: List, gold: List):
    shypothesis = set(hypothesis)
    sgold = set(gold)
    
    if len(shypothesis) == 0:
        return 1. if len(sgold) == 0 else 0.
    
    relevant = shypothesis.intersection(sgold)
    return len(relevant)/len(shypothesis)

def compute_recall(hypothesis: List, gold: List):
    shypothesis = set(hypothesis)
    sgold = set(gold)
    
    if len(sgold) == 0:
        return 1. if len(shypothesis) == 0 else 0.
    
    relevant = shypothesis.intersection(sgold)
    return len(relevant)/len(sgold)

def load_dataset(path: str):
    if path.endswith(('.parquet', '.parquet.gzip')):
        return pd.read_parquet(path, engine='auto')
    elif path.endswith('.json'):
        return pd.read_json(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Evaluation bench for LLM",
                                    description="Evaluate LLMs for SPARQL generation")
    parser.add_argument('-d', '--dataset', required=True, type=str, help="The path to the dataset.")
    parser.add_argument('-g', '--gold', required=True, type=str, help="The path to the gold dataset (dataset with answers).")
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

    if not os.path.exists(args.gold):
        raise FileNotFoundError(f"The gold dataset file not found with path: {args.gold}")

    df = load_dataset(args.dataset)
    df_gold = load_dataset(args.gold)
    
    df_no_gen_fail = df.drop(failed_generation_index(df))
    df_exec_timeout = df_no_gen_fail.loc[df_no_gen_fail['execution'] == 'timeout']
    df_exec_fail = df_no_gen_fail.loc[df_no_gen_fail['execution'].str.startswith('exception')]
    df_exec_empty = df_no_gen_fail.loc[df_no_gen_fail['execution'].isnull()]
    df_exec_to_eval = df_no_gen_fail.drop(df_exec_timeout.index).drop(df_exec_fail.index).drop(df_exec_empty.index)
    df_eval = eval_dataset(df_exec_to_eval)
    
    df_gold_exec_timeout = df_gold.loc[df_gold['execution'] == 'timeout']
    df_gold_exec_fail = df_gold.loc[df_gold['execution'].str.startswith('exception')]
    df_gold_exec_empty = df_gold.loc[df_gold['execution'].isnull()]
    df_gold_exec_to_eval = df_gold.drop(df_gold_exec_timeout.index).drop(df_gold_exec_fail.index).drop(df_gold_exec_empty.index)
    df_gold_eval = eval_dataset(df_gold_exec_to_eval, "gold_eval")
    
    df_merged_eval = df_eval.merge(df_gold_eval, how="left", left_index=True)
    df_merged_eval['precision'] = df_merged_eval.apply(lambda x: compute_precision(get_nested_values(x['eval']), get_nested_values(x['gold_eval'])))
    df_merged_eval['recall'] = df_merged_eval.apply(lambda x: compute_recall(get_nested_values(x['eval']), get_nested_values(x['gold_eval'])))
    
    m_precision = df_merged_eval['precision'].mean()
    m_recall = df_merged_eval['recall'].mean()
    m_fscore = 2*m_precision*m_recall/(m_precision+m_recall)

    bleu_score = corpus_bleu([[x.split()] for x in df_no_gen_fail['target']], [x.split() for x in df_no_gen_fail['translated_prompt']])
    meteor_score = corpus_meteor(df_no_gen_fail['target'], df_no_gen_fail['translated_prompt'])
    
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
                        "bleu_score": bleu_score,
                        "meteor_score": meteor_score,
                        "precision": m_precision,
                        "recall": m_recall,
                        "f1score": m_fscore
                    })
    
    os.makedirs(args.output, exist_ok=True)
    serie.to_json(os.path.join(args.output, f"{args.save_name}.json"))