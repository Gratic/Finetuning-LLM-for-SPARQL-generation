from ast import literal_eval
import logging
import pandas as pd
from typing import Union, Dict, List
from pathlib import Path

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

def load_dataset(dataset_path: Union[str, Path]):
    if not isinstance(dataset_path, str) and not isinstance(dataset_path, Path):
        raise TypeError("dataset_path must be a Path object or string.")
    
    p = dataset_path if isinstance(dataset_path, Path) else Path(dataset_path)
    if not p.exists():
        raise FileNotFoundError(f"The dataset was not found at: {str(p)}")
    
    filename = p.name
    path = p.absolute().resolve()
    
    if filename.endswith((".parquet.gzip", ".parquet")):
        try:
            return pd.read_parquet(path, engine="fastparquet")
        except:
            return pd.read_parquet(path)
    elif filename.endswith(".json"):
        return pd.read_json(path)
    elif filename.endswith(".pkl"):
        return pd.read_pickle(path)
    raise ValueError(f"The provided dataset format is not taken in charge. Use json, parquet or pickle. Found: {filename}")
    
def safe_loc(x, df, column, default=None):
    try:
        ans = df[[column]].loc[int(x.name)].item()
    except:
        try:
            ans = df[[column]].loc[str(x.name)].item()
        except:
            ans = default
    return ans