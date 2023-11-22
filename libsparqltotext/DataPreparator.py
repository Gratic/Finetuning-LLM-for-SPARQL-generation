import json
import pandas as pd
from collections.abc import Callable

class DataPreparator():
    def __init__(self, prompt_callback: Callable[[str, pd.Series], str], dataset_path: str, system_prompt: str, prompt_preparation: str) -> None:
        self.prompt_callback = prompt_callback
        self.dataset_path = dataset_path
        self.system_prompt = system_prompt
        self.prompt_preparation = prompt_preparation.lower()
        self.dataset = None
        self.data_prepared = False
    
    def load_and_prepare_queries(self) -> pd.DataFrame:
        cleaned_queries = None
        with open(self.dataset_path, 'r') as f:
            cleaned_queries = json.load(f)

        df_dataset = pd.DataFrame(cleaned_queries)
        self._verify_base_dataset_format(df_dataset)
        
        if self.prompt_preparation == "yes" or (self.prompt_preparation == "auto" and 'prompt' not in df_dataset.columns):
            df_dataset["prompt"] = df_dataset.apply(lambda x: self.prompt_callback(self.system_prompt, x), axis=1)
                
        df_dataset["result"] = df_dataset.apply(lambda x: None, axis=1)
        df_dataset["full_answer"] = df_dataset.apply(lambda x: None, axis=1)
        df_dataset["is_skipped"] = df_dataset.apply(lambda x: None, axis=1)
        df_dataset["is_prompt_too_long"] = df_dataset.apply(lambda x: None, axis=1)
        
        self._verify_after_processing_dataset_format(df_dataset)
        
        self.dataset = df_dataset
        self.data_prepared = True
        
        return self.dataset

    def get_dataset(self) -> pd.DataFrame:
        if not self.data_prepared:
            self.load_and_prepare_queries()
        return self.dataset
    
    @staticmethod
    def _verify_base_dataset_format(dataset: pd.DataFrame):
        columns = ["query", "context", "description"]
        for col in columns:
            if col not in dataset.columns:
                raise ValueError(f"Dataset does not contain the column {col}.")
    
    @staticmethod
    def _verify_after_processing_dataset_format(dataset: pd.DataFrame):
        columns = ["query", "context", "description", "result", "prompt", "full_answer", "is_skipped", "is_prompt_too_long"]
        for col in columns:
            if col not in dataset.columns:
                raise ValueError(f"Dataset does not contain the column {col}.")