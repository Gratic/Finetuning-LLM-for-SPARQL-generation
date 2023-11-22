import json
import pandas as pd
from collections.abc import Callable
from typing import List

def row_data_into_text(row: pd.Series) -> str:
  return f"QUERY=\"{row['query']}\" DESCRIPTION=\"{row['description']}\" CONTEXT=\"{row['context']}\""

def basic_prompt(sys_prompt: str, query: pd.Series) -> str:
  return f"""<s>[INST] {sys_prompt}
{row_data_into_text(query)}

User: Read QUERY, DESCRIPTION and CONTEXT. There is a machine capable of writing the given QUERY if we ask it the right prompt. Please do not include parts of QUERY in your answers. Give a list of 3 prompts that would give QUERY. [\INST]
Llama:"""

def load_and_prepare_queries(prompt_callback: Callable[[str, pd.Series], str], dataset_path: str, system_prompt: str, prompt_preparation: str) -> pd.DataFrame:
    with open(dataset_path, 'r') as f:
        cleaned_queries = json.load(f)

    df_dataset = pd.DataFrame(cleaned_queries)
    if prompt_preparation.lower() == "yes" or (prompt_preparation.lower() == "auto" and 'prompt' not in df_dataset.columns):
        df_dataset["prompt"] = df_dataset.apply(lambda x: prompt_callback(system_prompt, x), axis=1)
            
    df_dataset["result"] = df_dataset.apply(lambda x: None, axis=1)
    df_dataset["full_answer"] = df_dataset.apply(lambda x: None, axis=1)
    df_dataset["is_skipped"] = df_dataset.apply(lambda x: None, axis=1)
    df_dataset["is_prompt_too_long"] = df_dataset.apply(lambda x: None, axis=1)
    return df_dataset