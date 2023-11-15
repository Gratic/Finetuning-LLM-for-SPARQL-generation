import json
import pandas as pd


def row_data_into_text(row):
  return f"QUERY=\"{row['query']}\" DESCRIPTION=\"{row['description']}\" CONTEXT=\"{row['context']}\""

def basic_prompt(sys_prompt, query):
  return f"""<s>[INST] {sys_prompt}
{row_data_into_text(query)}

User: Read QUERY, DESCRIPTION and CONTEXT. There is a machine capable of writing the given QUERY if we ask it the right prompt. Please do not include parts of QUERY in your answers. Give a list of 3 prompts that would give QUERY. [\INST]
Llama:"""

def are_results_acceptable(results, banned_words):
    is_good_quality = True
    
    if len(results) == 0:
        return False
    
    for result in results:
        for word in banned_words:
            if word in result:
                is_good_quality = False
    return is_good_quality

def load_and_prepare_queries(prompt_callback, dataset_path, system_prompt, prompt_preparation):
    with open(dataset_path, 'r') as f:
        cleaned_queries = json.load(f)

    df_dataset = pd.DataFrame(cleaned_queries)
    if prompt_preparation.lower() == "yes" or (prompt_preparation.lower() == "auto" and 'prompt' not in df_dataset.columns):
        df_dataset["prompt"] = df_dataset.apply(lambda x: prompt_callback(system_prompt, x), axis=1)
            
    df_dataset["result"] = df_dataset.apply(lambda x: "", axis=1)
    df_dataset["full_answer"] = df_dataset.apply(lambda x: "", axis=1)
    return df_dataset

def prepare_request_payload(prompt, number_of_token_to_predict, temperature):
    payload = dict()
    payload["prompt"] = prompt
    payload["n_predict"] = number_of_token_to_predict
    payload["temperature"] = temperature

    return payload