import json
import pandas as pd
from collections.abc import Callable

def row_data_into_text(row: pd.Series) -> str:
  return f"QUERY=\"{row['query']}\" DESCRIPTION=\"{row['description']}\" CONTEXT=\"{row['context']}\""

def basic_prompt(sys_prompt: str, query: pd.Series) -> str:
  return f"""<s>[INST] {sys_prompt}
{row_data_into_text(query)}

User: Read QUERY, DESCRIPTION and CONTEXT. There is a machine capable of writing the given QUERY if we ask it the right prompt. Please do not include parts of QUERY in your answers. Give a list of 3 prompts that would give QUERY. [\INST]
Llama:"""