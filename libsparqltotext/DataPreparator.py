import json
import pandas as pd
from collections.abc import Callable

class DataPreparator():
    def __init__(self, prompt_callback: Callable[[str, pd.Series], str], system_prompt: str, prompt_preparation: str) -> None:
        self.prompt_callback = prompt_callback
        self.dataset_path = None
        self.system_prompt = system_prompt
        self.prompt_preparation = prompt_preparation.lower()
        self.raw_dataset = None
        self.dataset = None
        self.data_prepared = False
        self.data_loaded = False
    
    def prepare_dataset(self) -> pd.DataFrame:
        if not self.data_loaded:
            raise ValueError("The dataset is not loaded. Please use the load_dataframe() function to load a dataset.")
        
        if self.prompt_preparation == "yes" or (self.prompt_preparation == "auto" and 'prompt' not in self.dataset.columns):
            self.dataset["prompt"] = self.dataset.apply(lambda x: self.prompt_callback(self.system_prompt, x), axis=1)
                
        self.dataset["result"] = self.dataset.apply(lambda x: None, axis=1)
        self.dataset["full_answer"] = self.dataset.apply(lambda x: None, axis=1)
        self.dataset["is_skipped"] = self.dataset.apply(lambda x: None, axis=1)
        self.dataset["is_prompt_too_long"] = self.dataset.apply(lambda x: None, axis=1)
        
        self._verify_after_processing_dataset_format(self.dataset)
        
        self.data_prepared = True
        
        return self.dataset

    def load_dataframe(self, dataset_path: str):
        self.dataset_path = dataset_path
        
        cleaned_queries = None
        with open(self.dataset_path, 'r') as f:
            cleaned_queries = json.load(f)
            

        self.raw_dataset = pd.DataFrame(cleaned_queries)
        self._verify_base_dataset_format(self.raw_dataset)
        
        self.dataset = self.raw_dataset.copy()
        self.data_loaded = True
        return self.dataset

    def get_dataset(self) -> pd.DataFrame:
        if not self.data_prepared:
            self.prepare_dataset()
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