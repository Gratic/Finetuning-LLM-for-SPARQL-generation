from ast import literal_eval
from pathlib import Path
from typing import Union, Dict, List
import logging
import pandas as pd

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
    
def safe_loc(x, df, column, default=None, as_serie=False):
    if not as_serie:
        try:
            return df.loc[int(x.name)][[column]].item()
        except:
            try:
                return df.loc[str(x.name)][[column]].item()
            except:
                return default
    else:
        try:
            return df.loc[int(x.name)][column]
        except:
            try:
                return df.loc[str(x.name)][column]
            except:
                return default

# TODO: make test for this function
def series_or_dataframe_to_list(obj):
    if isinstance(obj, List):
        return obj
    
    if isinstance(obj, pd.Series):
        obj = obj.to_list()
    elif isinstance(obj, pd.DataFrame):
        if obj.empty:
            obj = []
        else:
            obj = obj[obj.columns[0]].to_list()
    elif obj == None:
        obj = None
    else:
        raise NotImplementedError(f"This case was not implemented, found: {type(obj)}")
    return obj

def get_value_from_sparql_key(data_dict):
    if "type" in data_dict.keys():
        ttype = data_dict["type"]
        
        if ttype == "uri":
            return data_dict["value"]
        
        elif ttype == "literal":
            if "datatype" in data_dict.keys() and data_dict['datatype'] == 'http://www.w3.org/2001/XMLSchema#integer':
                return int(data_dict["value"])
            
            return data_dict["value"]
        
        raise NotImplementedError(f"This type was not implemented, found: {ttype}.")
        
    raise ValueError("Type is not in data_dict.")

def get_columns_from_sparql_response(response):
    if not isinstance(response, list):
        raise Exception("The response needs to be evaluated or at least a List.")
    
    if len(response) == 0:
        return None
    
    return list(response[0].keys())

def make_dataframe_from_sparql_response(response):
    columns = get_columns_from_sparql_response(response)
    
    if columns == None:
        return pd.DataFrame()
    
    df = {k:[] for k in columns}
    
    for row in response:
        for k in df.keys():
            if k in row.keys():
                data = get_value_from_sparql_key(row[k])
            else:
                data = None
            df[k].append(data)
    
    return pd.DataFrame(data=df)

def set_seed(seed: int):
    import random
    import numpy as np
    import torch
    """
    Sets seeds across package dependencies for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
