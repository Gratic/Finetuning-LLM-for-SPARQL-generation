from ast import literal_eval
from nltk.translate.meteor_score import single_meteor_score
from typing import List, Dict, Union
import logging
import pandas as pd

def corpus_meteor(references: List, hypotheses: List):
    meteor_scores = 0.
    for ref, hyp in zip(references, hypotheses):
        meteor_scores += single_meteor_score(ref.split(), hyp.split())
    return meteor_scores / float(len(references))

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

def failed_generation_index(dataset: pd.DataFrame):
    return dataset.loc[dataset['has_error'] == True].index

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

def get_nested_values(element: Union[Dict, str, None]):
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
    elif element is None:
        values = []
    else:
        logging.error(f"get_nested_values doesn't have an implementation for: {type(element)}.")
        raise TypeError(f"Compatible types are Dict and List, found: {type(element)}.")
    return values

def load_dataset(path: str):
    if path.endswith(('.parquet', '.parquet.gzip')):
        return pd.read_parquet(path, engine='auto')
    elif path.endswith('.json'):
        return pd.read_json(path)
    
def safe_loc(x, df, column, default=None):
    try:
        ans = df[[column]].loc[str(x.name)]
    except:
        ans = default
    return ans