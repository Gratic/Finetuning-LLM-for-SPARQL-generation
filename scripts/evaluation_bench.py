import pandas as pd
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import single_meteor_score
import argparse
import os

def failed_generation_index(dataset: pd.DataFrame):
    return dataset.loc[dataset['has_error'] == True].index

def corpus_meteor(references, hypotheses):
    meteor_scores = 0.
    for ref, hyp in zip(references, hypotheses):
        meteor_scores += single_meteor_score(ref, hyp)
    return meteor_scores / float(len(references))

def eval_dataset(dataset, col_name = "eval"):
    df_eval = dataset.copy()
    df_eval[col_name] = df_eval.apply(lambda x: None, axis=1)
    for (i, row) in df_eval.iterrows():
        try:
            df_eval.at[i, col_name] = eval(row['execution'])
        except Exception as inst:
            print(inst)
    return df_eval[~df_eval[col_name].isnull()]

def get_values(element):
    values = []
    if isinstance(element, dict):
        for k, v in element.items():
            if isinstance(v, dict):
                values += get_values(v)
            elif isinstance(v, str):
                if 'value' in k:
                    values.append(v)
    if isinstance(element, list):
        for el in element:
            values += get_values(el)
    return values

def compute_precision(hyp, gold):
    sethyp = set(hyp)
    setgold = set(gold)
    
    relevant = sethyp.intersection(setgold)
    return len(relevant)/len(sethyp)

def compute_recall(hyp, gold):
    sethyp = set(hyp)
    setgold = set(gold)
    
    relevant = sethyp.intersection(setgold)
    return len(relevant)/len(setgold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Evaluation bench for LLM",
                                    description="Evaluate LLMs for SPARQL generation")
    parser.add_argument('-d', '--dataset', required=True, type=str, help="The path to the dataset.")
    parser.add_argument('-g', '--gold', required=True, type=str, help="The path to the gold dataset (dataset with answers).")
    parser.add_argument('-m', '--model', required=True, type=str, help="The model name.")
    parser.add_argument('-o', '--output', required=True, type=str, help="Folder to output the results.")
    parser.add_argument('-sn', '--save-name', required=True, type=str, help="Name of the save file.")

    args = parser.parse_args()

    df = pd.read_parquet(args.dataset)
    df_gold = pd.read_parquet(args.gold)
    
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
    
    df_merged_eval = df_eval.merge(how="left", left_index=True)
    df_merged_eval['precision'] = df_merged_eval.apply(lambda x: compute_precision(get_values(x['eval']), get_values(x['gold_eval'])))
    df_merged_eval['recall'] = df_merged_eval.apply(lambda x: compute_recall(get_values(x['eval']), get_values(x['gold_eval'])))
    
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

    serie.to_json(os.path.join([args.output, f"{args.save_name}.json"]))