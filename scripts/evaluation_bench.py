import sys
from pathlib import Path
sys.path.append(Path("modules").absolute().__str__())

from data_utils import eval_dataset, get_nested_values, load_dataset, safe_loc, make_dataframe_from_sparql_response
from evaluation_utils import  is_correct_SPARQL_query, keep_id_columns, cross_product_func, precision_recall_fscore_support_wrapper, average_precision_wrapper
import argparse
import json
import logging
import nltk
import os
import pandas as pd
import evaluate

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
    df_no_gen_fail = df # df.drop(failed_generation_index(df))
    df_exec_timeout = df_no_gen_fail.loc[df_no_gen_fail['execution'] == 'timeout']
    df_exec_fail = df_no_gen_fail.loc[df_no_gen_fail['execution'].str.startswith('exception')]
    df_exec_empty = df_no_gen_fail.loc[df_no_gen_fail['execution'].isnull()]
    df_exec_to_eval = df_no_gen_fail.drop(df_exec_timeout.index).drop(df_exec_fail.index).drop(df_exec_empty.index)
    df_eval = eval_dataset(df_exec_to_eval)
    df_eval['get_nested_values'] = df_eval.apply(lambda x: get_nested_values(x['eval']), axis=1)
    df_eval['eval_df'] = df_eval.apply(lambda x: make_dataframe_from_sparql_response(x['eval']), axis=1)
    df_eval['id_columns'] = df_eval.apply(lambda x: keep_id_columns(x['eval_df']), axis=1)
    
    df_gold_eval = None
    if args.gold != None:
        df_gold = load_dataset(args.gold)
        df_gold_exec_timeout = df_gold.loc[df_gold['execution'] == 'timeout']
        df_gold_exec_fail = df_gold.loc[df_gold['execution'].str.startswith('exception')]
        df_gold_exec_empty = df_gold.loc[df_gold['execution'].isnull()]
        df_gold_exec_to_eval = df_gold.drop(df_gold_exec_timeout.index).drop(df_gold_exec_fail.index).drop(df_gold_exec_empty.index)
        df_gold_eval = eval_dataset(df_gold_exec_to_eval, "gold_eval")
        df_gold_eval['gold_get_nested_values'] = df_gold_eval.apply(lambda x: get_nested_values(x['gold_eval']), axis=1)
        df_gold_eval['gold_eval_df'] = df_gold_eval.apply(lambda x: make_dataframe_from_sparql_response(x['gold_eval']), axis=1)
        df_gold_eval['gold_id_columns'] = df_gold_eval.apply(lambda x: keep_id_columns(x['gold_eval_df']), axis=1)
    else:
        with open(args.preprocess_gold, "r") as f:
            data = json.load(f)
        df_gold_eval = pd.read_json(data['df_gold_eval'])
        df_gold_exec_timeout = df_gold_eval.loc[df_gold_eval['execution'] == 'timeout']
        df_gold_exec_fail = df_gold_eval.loc[df_gold_eval['execution'].str.startswith('exception')]
        df_gold_exec_empty = df_gold_eval.loc[df_gold_eval['execution'].isnull()]
        df_gold_exec_to_eval = df_gold_eval.drop(df_gold_exec_timeout.index).drop(df_gold_exec_fail.index).drop(df_gold_exec_empty.index)
        
    
    df_merged_eval = df_eval.copy()
    
    # Merging manually
    df_merged_eval["gold_eval"] = df_merged_eval.apply(lambda x: safe_loc(x, df_gold_eval, "gold_eval", default=None), axis=1)
    df_merged_eval["gold_get_nested_values"] = df_merged_eval.apply(lambda x: safe_loc(x, df_gold_eval, "gold_get_nested_values", default=[]), axis=1)
    df_merged_eval["gold_eval_df"] = df_merged_eval.apply(lambda x: safe_loc(x, df_gold_eval, "gold_eval_df", default=pd.DataFrame()), axis=1)
    df_merged_eval["gold_id_columns"] = df_merged_eval.apply(lambda x: safe_loc(x, df_gold_eval, "gold_id_columns", default=pd.DataFrame()), axis=1)
    
    # Computing metrics using scikit-learn

    df_merged_eval['get_nested_values_precision_recall_fscore'] = df_merged_eval.apply(lambda x: precision_recall_fscore_support_wrapper(
        x['gold_get_nested_values'],
        x['get_nested_values']
    ), axis=1)

    df_merged_eval['cross_precision_recall_fscore'] = df_merged_eval.apply(lambda x: cross_product_func(
        func=precision_recall_fscore_support_wrapper,
        y_true=x['gold_eval_df'].apply(lambda y: y.fillna(value="")),
        y_pred=x['eval_df'].apply(lambda y: y.fillna(value="")),
        maximization=True,
        use_binarizer=True,
        average="samples"
    )
    , axis=1)

    df_merged_eval['id_precision_recall_fscore'] = df_merged_eval.apply(lambda x: cross_product_func(
        func=precision_recall_fscore_support_wrapper,
        y_true=x['gold_id_columns'].apply(lambda y: y.fillna(value="")),
        y_pred=x['id_columns'].apply(lambda y: y.fillna(value="")),
        maximization=True,
        use_binarizer=True,
        average="samples"
    )
    , axis=1)
    
    # Computing average precision with custom function
    df_merged_eval['get_nested_values_average_precision'] = df_merged_eval.apply(lambda x: average_precision_wrapper(
        y_true=x['gold_get_nested_values'],
        y_pred=x['get_nested_values']
    ), axis=1)

    df_merged_eval['cross_average_precision'] = df_merged_eval.apply(lambda x: cross_product_func(
        func=average_precision_wrapper,
        y_true=x['gold_eval_df'].apply(lambda y: y.fillna(value="")),
        y_pred=x['eval_df'].apply(lambda y: y.fillna(value="")),
        maximization=True,
    )
    , axis=1)

    df_merged_eval['id_average_precision'] = df_merged_eval.apply(lambda x: cross_product_func(
        func=average_precision_wrapper,
        y_true=x['gold_id_columns'].apply(lambda y: y.fillna(value="")),
        y_pred=x['id_columns'].apply(lambda y: y.fillna(value="")),
        maximization=True,
    )
    , axis=1)
    
    gnv_precision = df_merged_eval['get_nested_values_precision_recall_fscore'].map(lambda r: r[0] if isinstance(r, tuple) else 0).mean()
    gnv_recall = df_merged_eval['get_nested_values_precision_recall_fscore'].map(lambda r: r[1] if isinstance(r, tuple) else 0).mean()
    gnv_fscore = df_merged_eval['get_nested_values_precision_recall_fscore'].map(lambda r: r[2] if isinstance(r, tuple) else 0).mean()

    cross_precision = df_merged_eval['cross_precision_recall_fscore'].map(lambda r: r[0] if isinstance(r, tuple) else 0).mean()
    cross_recall = df_merged_eval['cross_precision_recall_fscore'].map(lambda r: r[1] if isinstance(r, tuple) else 0).mean()
    cross_fscore = df_merged_eval['cross_precision_recall_fscore'].map(lambda r: r[2] if isinstance(r, tuple) else 0).mean()

    id_precision = df_merged_eval['id_precision_recall_fscore'].map(lambda r: r[0] if isinstance(r, tuple) else 0).mean()
    id_recall = df_merged_eval['id_precision_recall_fscore'].map(lambda r: r[1] if isinstance(r, tuple) else 0).mean()
    id_fscore = df_merged_eval['id_precision_recall_fscore'].map(lambda r: r[2] if isinstance(r, tuple) else 0).mean()

    gnv_map = df_merged_eval['get_nested_values_average_precision'].mean()
    cross_map = df_merged_eval['cross_average_precision'].mean()
    id_map = df_merged_eval['id_average_precision'].mean()

    nltk.download("punkt", quiet=True)
    rouge_metric = evaluate.load("rouge")
    bleu_metric = evaluate.load("bleu")
    meteor_metric = evaluate.load("meteor")

    decoded_labels = df_merged_eval['target_raw'].map(lambda x: "\n".join(nltk.sent_tokenize(x.strip()))).to_list()
    decoded_preds = df_merged_eval['output'].map(lambda x: "\n".join(nltk.sent_tokenize(x.strip()))).to_list()

    rouge_dict = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # bleu_score = corpus_bleu([[x.split()] for x in df['target_template']], [x.split() for x in df['translated_prompt']])
    bleu_dict = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    # meteor_dict = corpus_meteor(hypotheses=decoded_preds, references=decoded_labels)
    meteor_dict = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)
    correct_syntax = sum(list(map(lambda y: int(y[1]), df.apply(lambda x: is_correct_SPARQL_query(x['translated_prompt']), axis=1).items()))) / len(df)

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
        "gold_num_rows": len(df_gold_eval),
        "gold_num_exec_timeout": len(df_gold_exec_timeout),
        "gold_num_exec_fail": len(df_gold_exec_fail),
        "gold_num_exec_empty": len(df_gold_exec_empty),
        "gold_num_exec_to_eval": len(df_gold_exec_to_eval),
        "gold_num_eval_empty": len(df_gold_eval.loc[df_gold_eval['gold_eval'].map(len) == 0]),
        "bleu_score": bleu_dict["bleu"],
        "meteor_score": meteor_dict['meteor'],
        **rouge_dict,
        "get_nested_values_precision": gnv_precision,
        "get_nested_values_recall": gnv_recall,
        "get_nested_values_f1score": gnv_fscore,
        "get_nested_values_mean_average_precision": gnv_map,
        "id_precision": id_precision,
        "id_recall": id_recall,
        "id_f1score": id_fscore,
        "id_mean_average_precision": id_map,
        "cross_precision": cross_precision,
        "cross_recall": cross_recall,
        "cross_f1score": cross_fscore,
        "cross_mean_average_precision": cross_map,
        "correct_syntax": correct_syntax,
    })
    
    os.makedirs(args.output, exist_ok=True)
    serie.to_json(os.path.join(args.output, f"{args.save_name}.json"))