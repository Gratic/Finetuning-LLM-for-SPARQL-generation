from nltk.translate.meteor_score import single_meteor_score
from typing import List
from SPARQL_parser import SPARQL
import re

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