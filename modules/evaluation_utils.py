from data_utils import load_dataset, eval_dataset, failed_generation_index, safe_loc, make_dataframe_from_sparql_response, get_nested_values
from itertools import product
from nltk.translate.meteor_score import single_meteor_score
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from SPARQL_parser import SPARQL
from typing import List
import collections
import ir_measures
import json
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
    if not isinstance(query, str):
        return False
    
    if query.strip() == "":
        return False
    
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

def cross_product_func(func, y_true:pd.DataFrame, y_pred:pd.DataFrame, maximization:bool=True, **func_args):
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
    
    is_first = True
    
    for x, y in product(y_true.columns.to_list(), y_pred.columns.to_list()):
        labels = y_true[x].loc[y_true[x] != ''].to_list() # drop empty values
        preds = y_pred[y].loc[y_pred[y] != ''].to_list()
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

def load_and_merge_evaluation_and_gold_dataset(args):
    df, df_exec_timeout, df_exec_fail, df_exec_empty, df_exec_to_eval, df_eval = process_dataset_for_evaluation(args.dataset)
    
    df_gold_eval = None
    if 'gold' in args and args.gold != None:
        df_gold, df_gold_exec_timeout, df_gold_exec_fail, df_gold_exec_empty, df_gold_exec_to_eval, df_gold_eval = process_dataset_for_evaluation(args.gold, prefix="gold_")
    else:
        with open(args.preprocess_gold, "r") as f:
            data = json.load(f)
            
        df_gold = pd.read_json(data['df_gold'])
        df_gold_exec_timeout = pd.read_json(data['df_gold_exec_timeout'])
        df_gold_exec_fail = pd.read_json(data['df_gold_exec_fail'])
        df_gold_exec_empty = pd.read_json(data['df_gold_exec_empty'])
        df_gold_exec_to_eval = pd.read_json(data['df_gold_exec_to_eval'])
        df_gold_eval = pd.read_json(data['df_gold_eval'])
        
        # When data is serialized into json, data that is pd.DataFrame becomes Dict.
        # We must convert it back from Dict to pd.DataFrame
        df_gold_eval['gold_eval_df'] = df_gold_eval.apply(lambda x: pd.DataFrame(data=x['gold_eval_df']), axis=1)
        df_gold_eval['gold_id_columns'] = df_gold_eval.apply(lambda x: pd.DataFrame(data=x['gold_id_columns']), axis=1)
    
    df_merged_eval = df_eval.copy()
    
    # Merging manually
    df_merged_eval["gold_eval"] = df_merged_eval.apply(lambda x: safe_loc(x, df_gold_eval, "gold_eval", default=None), axis=1)
    df_merged_eval["gold_get_nested_values"] = df_merged_eval.apply(lambda x: safe_loc(x, df_gold_eval, "gold_get_nested_values", default=[]), axis=1)
    df_merged_eval["gold_eval_df"] = df_merged_eval.apply(lambda x: safe_loc(x, df_gold_eval, "gold_eval_df", default=pd.DataFrame()), axis=1)
    df_merged_eval["gold_id_columns"] = df_merged_eval.apply(lambda x: safe_loc(x, df_gold_eval, "gold_id_columns", default=pd.DataFrame()), axis=1)
    return df,df_exec_timeout,df_exec_fail,df_exec_empty,df_exec_to_eval,df_eval,df_gold_eval,df_gold_exec_timeout,df_gold_exec_fail,df_gold_exec_empty,df_gold_exec_to_eval,df_merged_eval

def process_dataset_for_evaluation(dataset, prefix=""):
    df = load_dataset(dataset)
    df_no_gen_fail = df.drop(failed_generation_index(df))
    df_exec_timeout = df_no_gen_fail.loc[df_no_gen_fail['execution'] == 'timeout']
    df_exec_fail = df_no_gen_fail.loc[df_no_gen_fail['execution'].str.startswith('exception')]
    df_exec_empty = df_no_gen_fail.loc[df_no_gen_fail['execution'].str.startswith('[]')]
    df_exec_to_eval = df_no_gen_fail.drop(df_exec_timeout.index).drop(df_exec_fail.index).drop(df_exec_empty.index)
    df_eval = eval_dataset(df_exec_to_eval, col_name=f"{prefix}eval")
    df_eval[f'{prefix}get_nested_values'] = df_eval.apply(lambda x: get_nested_values(x[f'{prefix}eval']), axis=1)
    df_eval[f'{prefix}eval_df'] = df_eval.apply(lambda x: make_dataframe_from_sparql_response(x[f'{prefix}eval']), axis=1)
    df_eval[f'{prefix}id_columns'] = df_eval.apply(lambda x: keep_id_columns(x[f'{prefix}eval_df']), axis=1)
    return df,df_exec_timeout,df_exec_fail,df_exec_empty,df_exec_to_eval,df_eval

def make_qrel_namedtuple_from_element(qid:str, element, relevance:int):
    if not isinstance(relevance, int):
        raise TypeError()
    if relevance != 0 and relevance != 1:
        raise ValueError()
    
    value = None
    if isinstance(element, str):
        value = element
    elif isinstance(element, int) or isinstance(element, float):
        value = str(element)
    else:
        raise NotImplementedError()
    return ir_measures.Qrel(query_id=qid, doc_id=value, relevance=relevance)

def transform_df_column_into_qrel_list(qid:str, column: pd.Series):
    return column.map(lambda x: make_qrel_namedtuple_from_element(qid, x, 1)).to_list()

def transform_list_into_qrel_list(qid:str, column: list):
    return [make_qrel_namedtuple_from_element(qid, x, 1) for x in filter(lambda x: x != None, column)]

def transform_df_into_qrel_list(df: pd.DataFrame, output:str="aggregated"):
    if not isinstance(output, str):
        raise TypeError()
    if output not in ['aggregated', 'list_of_list']:
        raise ValueError()
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError()
    
    if df.empty:
        return []
    
    columns = list(df.columns)
    results = []
    for col in columns:
        if output == "aggregated":
            results += transform_df_column_into_qrel_list(qid="0", column=df[col])
        elif output == "list_of_list":
            results.append(transform_df_column_into_qrel_list(qid="0", column=df[col]))
            
    return results

def make_scoreddoc_namedtuple_from_element(qid:str, element, score:float):
    if not isinstance(score, int) and not isinstance(score, float):
        raise TypeError()
    
    value = None
    if isinstance(element, str):
        value = element
    elif isinstance(element, int) or isinstance(element, float):
        value = str(element)
    else:
        raise NotImplementedError()
    return ir_measures.ScoredDoc(query_id=qid, doc_id=value, score=score)

def transform_df_column_into_run_list(qid:str, column: pd.Series):
    return column.map(lambda x: make_scoreddoc_namedtuple_from_element(qid, x, 1)).to_list()

def transform_list_into_run_list(qid:str, column: list):
    return [make_scoreddoc_namedtuple_from_element(qid, x, 1) for x in filter(lambda x: x != None, column)]

def transform_df_into_run_list(df: pd.DataFrame, output:str="aggregated"):
    if not isinstance(output, str):
        raise TypeError()
    if output not in ['aggregated', 'list_of_list']:
        raise ValueError()
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError()
    
    if df.empty:
        return []
    
    columns = list(df.columns)
    results = []
    for col in columns:
        if output == "aggregated":
            results += transform_df_column_into_run_list(qid="0", column=df[col])
        elif output == "list_of_list":
            results.append(transform_df_column_into_run_list(qid="0", column=df[col]))
            
    return results

def cross_ir_measures_wrapper(y_true, y_pred, f):
    return f(transform_list_into_qrel_list(qid="0", column=y_true), transform_list_into_run_list(qid="0", column=y_pred))

ComputeMetricsResult = collections.namedtuple('ComputeMetricsResult', ['mean_average_precision', 'precision_k', 'recall_k', 'mean_reciprocal_rank', 'k'])
def compute_metrics_for_two_df(results, gold, k:int=5):
    if not isinstance(results, pd.DataFrame):
        raise TypeError()
    if not isinstance(gold, pd.DataFrame):
        raise TypeError()
    
    if not isinstance(k, int):
        raise TypeError()
    
    if results.empty or gold.empty:
        return ComputeMetricsResult(0., 0., 0., 0., k)
    
    results_dict = dict()
    results_dict['k'] = k
    results_dict['mean_average_precision'] = cross_product_func(func=cross_ir_measures_wrapper, y_true=gold, y_pred=results, f=ir_measures.AP.calc_aggregate)
    results_dict['precision_k'] = cross_product_func(func=cross_ir_measures_wrapper, y_true=gold, y_pred=results, f=ir_measures.parse_measure(f'P@{k}').calc_aggregate)
    results_dict['recall_k'] = cross_product_func(func=cross_ir_measures_wrapper, y_true=gold, y_pred=results, f=ir_measures.parse_measure(f'Success@{k}').calc_aggregate)
    results_dict['mean_reciprocal_rank'] = cross_product_func(func=cross_ir_measures_wrapper, y_true=gold, y_pred=results, f=ir_measures.parse_measure(f'RR@{k}').calc_aggregate)
        
    return ComputeMetricsResult(
        mean_average_precision=results_dict['mean_average_precision'],
        precision_k=results_dict['precision_k'],
        recall_k=results_dict['recall_k'],
        mean_reciprocal_rank=results_dict['mean_reciprocal_rank'],
        k=results_dict['k']
    )

def compute_metrics_for_two_list(results, gold, k:int=5):
    if not isinstance(results, list):
        raise TypeError()
    if not isinstance(gold, list):
        raise TypeError()
    
    if not isinstance(k, int):
        raise TypeError()
    
    if len(results) == 0 or len(gold) == 0:
        return ComputeMetricsResult(0., 0., 0., 0., k)
    
    qrels = transform_list_into_qrel_list(qid="0", column=gold)
    runs = transform_list_into_run_list(qid="0", column=results)
    
    results_dict = dict()
    results_dict['k'] = k
    results_dict['mean_average_precision'] = ir_measures.AP.calc_aggregate(qrels=qrels, run=runs)
    results_dict['precision_k'] = ir_measures.parse_measure(f'P@{k}').calc_aggregate(qrels=qrels, run=runs)
    results_dict['recall_k'] = ir_measures.parse_measure(f'Success@{k}').calc_aggregate(qrels=qrels, run=runs)
    results_dict['mean_reciprocal_rank'] = ir_measures.parse_measure(f'RR@{k}').calc_aggregate(qrels=qrels, run=runs)
        
    return ComputeMetricsResult(
        mean_average_precision=results_dict['mean_average_precision'],
        precision_k=results_dict['precision_k'],
        recall_k=results_dict['recall_k'],
        mean_reciprocal_rank=results_dict['mean_reciprocal_rank'],
        k=results_dict['k']
    )