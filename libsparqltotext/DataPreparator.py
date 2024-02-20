import json
import pandas as pd
from collections.abc import Callable
from .Provider import BaseProvider
from .utils import replace_from_dict, row_data_into_text

class DataPreparator():
    def __init__(self, provider: BaseProvider, template: str, system_prompt: str, prompt: str, lead_answer_prompt: str, prompt_preparation: str) -> None:
        self.provider = provider
        self.dataset_path = None
        self.template = template
        self.system_prompt = system_prompt
        self.prompt = prompt
        self.lead_answer_prompt = lead_answer_prompt
        self.prompt_preparation = prompt_preparation.lower()
        self.raw_dataset = None
        self.dataset = None
        self.data_prepared = False
        self.data_loaded = False
    
    def prepare_dataset(self) -> pd.DataFrame:
        if not self.data_loaded:
            raise ValueError("The dataset is not loaded. Please use the load_dataframe() function to load a dataset.")
        
        if self.prompt_preparation == "yes":
            self.dataset["prompt"] = self._prepare_prompts()
            self.dataset["num_tokens"] = self._prepare_num_tokens()
        elif self.prompt_preparation == "auto":
            if 'prompt' not in self.dataset.columns:
                self.dataset["prompt"] = self._prepare_prompts()
            if 'num_tokens' not in self.dataset.columns:
                self.dataset["num_tokens"] = self._prepare_num_tokens()
                
                
                
        self.dataset["result"] = self.dataset.apply(lambda x: None, axis=1)
        self.dataset["full_answer"] = self.dataset.apply(lambda x: None, axis=1)
        self.dataset["is_skipped"] = self.dataset.apply(lambda x: None, axis=1)
        self.dataset["is_prompt_too_long"] = self.dataset.apply(lambda x: None, axis=1)
        
        self._verify_after_processing_dataset_format(self.dataset)
        
        self.data_prepared = True
        
        return self.dataset

    def _prepare_num_tokens(self):
        return self.dataset.apply(lambda x: len(self.provider.get_tokens(x['prompt'])), axis=1)
    
    def _prepare_prompts(self):
        return self.dataset.apply(lambda x: replace_from_dict(self.template, {"system_prompt": self.system_prompt, "data": row_data_into_text(x), "prompt": self.prompt, "lead_answer_prompt": self.lead_answer_prompt}), axis=1)
        

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
        columns = ["query", "context", "description", "prompt", "num_tokens", "result", "full_answer", "is_skipped", "is_prompt_too_long"]
        for col in columns:
            if col not in dataset.columns:
                raise ValueError(f"Dataset does not contain the column {col}.")