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
    """
    Compute the precision metric for a given hypothesis and gold standard.
    
    If the hypothesis list is empty but also the gold then it will return 1, otherwise 0.
    """
    shypothesis = set(hypothesis) if hypothesis != None else set()
    sgold = set(gold) if gold != None else set()
    
    if len(shypothesis) == 0:
        return 1. if len(sgold) == 0 else 0.
    
    relevant = shypothesis.intersection(sgold)
    return len(relevant)/len(shypothesis)

def compute_recall(hypothesis: List, gold: List):
    """
    Compute the recall metric for a given hypothesis and gold standard.
    
    If the gold list is empty but also the hypothesis then it will return 1, otherwise 0.
    """
    shypothesis = set(hypothesis) if hypothesis != None else set()
    sgold = set(gold) if gold != None else set()
    
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
    """
    Recursively walk through a dictionary searching for every 'value' keys.
    Each 'value' key's value is appended to a list and then returned.
    
    If given a list, results of this function on each element of the list will be concatened and returned.
    
    An None element will return an empty list.
    
    If element is not a Dict, List or None, a Type Error is raised.
    """
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
    else:
        raise NotImplementedError(f"Filetype provided not supported, found: {path}")
    
def safe_loc(x, df, column, default=None):
    try:
        ans = df[[column]].loc[int(x.name)].item()
    except:
        try:
            ans = df[[column]].loc[str(x.name)].item()
        except:
            ans = default
    return ans

def average_precision(hyp, gold, k_max = None):
    if hyp == None or gold == None:
        return 0.
    
    k = len(hyp) if hyp != None else 0
    n = len(gold)

    if k_max != None:
        k = min(k_max, k)
        n = min(k_max, n)

    if n == 0:
        return 0.
    
    sumAp = 0
    prec_sum = 0.
    total_prec = 0.
    
    for i in range(k):
        total_prec += 1
        if hyp[i] in gold[:n]:
           prec_sum += 1
           sumAp += prec_sum/total_prec
    
    # return sum([compute_precision(hyp[:1+i], gold) * (1 if hyp[i] in gold else 0) for i in range(k)])/n
    return sumAp/n

def average_precision_slow(hyp, gold, max_k = 3):
    if hyp == None or gold == None:
        return 0.
    
    k = min(len(hyp), max_k)
    n = float(len(gold))
    
    if n == 0:
        return 0.
    
    return (sum([compute_precision(hyp[:1+i], gold) * (1 if hyp[i] in gold else 0) for i in range(k)]) if k > 0 else 0.)/n