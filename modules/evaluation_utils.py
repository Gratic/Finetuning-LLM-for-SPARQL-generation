from itertools import product
from nltk.translate.meteor_score import single_meteor_score
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from SPARQL_parser import SPARQL
from typing import List
import pandas as pd
import re
import warnings

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
    if hypothesis == None or gold == None:
        return 0
    
    if len(hypothesis) == 0 or len(gold) == 0:
        return 0
    
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
    if hypothesis == None or gold == None:
        return 0
    
    if len(hypothesis) == 0 or len(gold) == 0:
        return 0
    
    shypothesis = set(hypothesis) if hypothesis != None else set()
    sgold = set(gold) if gold != None else set()
    
    if len(sgold) == 0:
        return 1. if len(shypothesis) == 0 else 0.
    
    relevant = shypothesis.intersection(sgold)
    return len(relevant)/len(sgold)

def average_precision(hyp, gold, k_max = None):
    if (hyp == None or gold == None
    or len(hyp) == 0 or len(gold) == 0):
        return 0.
    
    n = len(gold)
    k = min(n, len(hyp))

    if k_max != None:
        k = min(k_max, k)
    
    sumAp = 0
    prec_sum = 0.
    total_prec = 0.
    
    for i in range(k):
        total_prec += 1
        if hyp[i] in gold[:n]:
           prec_sum += 1
           sumAp += prec_sum/total_prec
    
    return sumAp/n

def average_precision_slow(hyp, gold, max_k = 3):
    if hyp == None or gold == None:
        return 0.
    
    k = min(len(hyp), max_k)
    n = float(len(gold))
    
    if n == 0:
        return 0.
    
    return (sum([compute_precision(hyp[:1+i], gold) * (1 if hyp[i] in gold else 0) for i in range(k)]) if k > 0 else 0.)/n

def is_correct_SPARQL_query(query):
    query = re.sub(r"PREFIX \w+:.*\n", "", query)
    
    try:
        SPARQL(query)
    except:
        return False
    return True

def is_correct_SPARQL_query_for_parallel(x):
    from SPARQL_parser import SPARQL
    import re
    
    try:
        SPARQL(re.sub(r"PREFIX \w+:.*\n", "", x['query']))
    except:
        return False
    return True

def unique_metric(column: pd.Series):
    return len(column.unique())/len(column)

def is_entity_column(column: pd.Series):
    if not isinstance(column[0], str):
        return False
    # return all(column.str.lower().str.startswith("http://www.wikidata.org/entity/"))
    return all(column.str.lower().str.startswith("http://www.wikidata.org/"))

# def old_find_id_column(response_df):
#     if not isinstance(response_df, pd.DataFrame):
#         raise TypeError("response_df must be a pandas DataFrame.")
    
#     if response_df.empty:
#         return None
    
#     potential_id_columns = response_df.columns
    
#     if len(potential_id_columns) == 1:
#         return potential_id_columns[0]
    
#     unique_scores = [(column, unique_metric(response_df[column])) for column in response_df.columns]
#     unique_scores.sort(key=lambda x: x[1], reverse=True)
    
#     potential_id_columns = list(map(lambda x: x[0], filter(lambda x: x[1] == unique_scores[0][1], unique_scores)))
    
#     if len(potential_id_columns) == 1:
#         return potential_id_columns[0]
    
#     potential_id_columns_with_id = list(filter(lambda x: x.lower().startswith('id') or x.lower().endswith('id'), potential_id_columns))
#     if len(potential_id_columns_with_id) > 0:
#         potential_id_columns = potential_id_columns_with_id
    
#     if len(potential_id_columns) == 1:
#         return potential_id_columns[0]
    
#     entity_columns = list(filter(lambda x: is_entity_column(response_df[x]), potential_id_columns))
#     if len(entity_columns) > 0:
#         potential_id_columns = entity_columns
    
#     return potential_id_columns[0]

def keep_id_columns(response_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(response_df, pd.DataFrame):
        raise TypeError("response_df must be a pandas DataFrame.")
    
    if response_df.empty:
        return pd.DataFrame()
    
    potential_id_columns = response_df.columns.to_list()
    
    if len(potential_id_columns) == 1:
        return response_df
    
    entity_columns = list(filter(lambda x: is_entity_column(response_df[x]), potential_id_columns))
    if len(entity_columns) > 0:
        return response_df[entity_columns]
    
    unique_scores = [(column, unique_metric(response_df[column])) for column in response_df.columns]
    unique_scores.sort(key=lambda x: x[1], reverse=True)
    
    potential_id_columns = list(map(lambda x: x[0], filter(lambda x: x[1] == unique_scores[0][1], unique_scores)))
    
    if len(potential_id_columns) == 1:
        return response_df[potential_id_columns]
    
    potential_id_columns_with_id = list(filter(lambda x: x.lower().startswith('id') or x.lower().endswith('id'), potential_id_columns))
    if len(potential_id_columns_with_id) > 0:
        potential_id_columns = potential_id_columns_with_id
    
    return response_df[potential_id_columns]

def cross_product_func(func, y_true:pd.DataFrame, y_pred:pd.DataFrame, maximization:bool=True, use_binarizer:bool=False, **func_args):
    if isinstance(y_true, pd.Series):
        if y_true.empty:
            return 0. if maximization else 1.
        y_true = y_true.to_frame()
    if isinstance(y_pred, pd.Series):
        if y_pred.empty:
            return 0. if maximization else 1.
        y_pred = y_pred.to_frame()
    
    if not isinstance(y_true, pd.DataFrame):
        if y_true == None:
            return 0
        raise TypeError(f"This function requires both y_true and y_pred to be pandas DataFrame, found y_true= {type(y_true).__name__}")
    if not isinstance(y_pred, pd.DataFrame):
        if y_pred == None:
            return 0
        raise TypeError(f"This function requires both y_true and y_pred to be pandas DataFrame, found y_pred= {type(y_pred).__name__}")
    
    warnings.filterwarnings(action='ignore', category=UserWarning)
    
    result = 0. if maximization else 1.
    binarizer = MultiLabelBinarizer()
    
    is_first = True
    
    for x, y in product(y_true.columns.to_list(), y_pred.columns.to_list()):
        labels = y_true[x].to_list()
        preds = y_pred[y].to_list()
        try:
            res = func(labels, preds, **func_args)
        
        except Exception as inst:
            print(f"{x=}")
            print(f"{y=}")
            print(f"{labels=}")
            print(f"{preds=}")
            print(f"{len(labels)=}")
            print(f"{len(preds)=}")
            print(f"{y_true=}")
            print(f"{y_pred=}")
            raise inst
        
        if isinstance(res, tuple):
            if is_first:
                result = tuple([int(not maximization)] * (len(res) - 1))
                is_first = False

            result = tuple([max(res[i], result[i]) if maximization else min(res[i], result[i]) for i in range(len(res) - 1)])
        else:
            result = max(res, result) if maximization else min(res, result)
        
    warnings.filterwarnings(action='default', category=UserWarning)
    return result

def precision_recall_fscore_support_wrapper(y_true, y_pred, average='samples', zero_division=0.0):
    if len(y_true) == 0:
        return 0
    
    warnings.filterwarnings(action='ignore', category=UserWarning)

    binarizer = MultiLabelBinarizer()
    binarizer.fit([y_true])
    
    labels = binarizer.transform([y_true])
    
    if len(labels) != 0 and len(labels[0]) < 2:
        average = 'macro'
        
    results = precision_recall_fscore_support(
        y_true = labels,
        y_pred = binarizer.transform([y_pred]),
        average=average,
        zero_division=zero_division
    )
    
    warnings.filterwarnings(action='default', category=UserWarning)
    return results

def average_precision_wrapper(y_true, y_pred):
        
    results = average_precision(
        gold = y_true,
        hyp = y_pred,
    )
    
    return results