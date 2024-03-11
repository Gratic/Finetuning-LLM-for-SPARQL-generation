import json
import pandas as pd
from collections.abc import Callable
from .Provider import BaseProvider
from .utils import replace_from_dict, row_data_into_text
from modules.data_utils import load_dataset

class DataPreparator():
    def __init__(self, provider: BaseProvider, template: str, system_prompt: str, prompt: str, lead_answer_prompt: str, prompt_preparation: str, query_column:str="query", prefix:str="basic_") -> None:
        self.provider = provider
        self.dataset_path = None
        self.template = template
        self.system_prompt = system_prompt
        self.prompt = prompt
        self.lead_answer_prompt = lead_answer_prompt
        self.prompt_preparation = prompt_preparation.lower()
        self.prefix = prefix
        self.query_column = query_column
        self.raw_dataset = None
        self.dataset = None
        self.data_prepared = False
        self.data_loaded = False
    
    def prepare_dataset(self) -> pd.DataFrame:
        if not self.data_loaded:
            raise ValueError("The dataset is not loaded. Please use the load_dataframe() function to load a dataset.")
        
        if self.prompt_preparation == "yes":
            self.dataset[f"{self.prefix}prompt"] = self._prepare_prompts()
            self.dataset[f"{self.prefix}num_tokens"] = self._prepare_num_tokens()
        elif self.prompt_preparation == "auto":
            if f'{self.prefix}prompt' not in self.dataset.columns:
                self.dataset[f"{self.prefix}prompt"] = self._prepare_prompts()
            if f'{self.prefix}num_tokens' not in self.dataset.columns:
                self.dataset[f"{self.prefix}num_tokens"] = self._prepare_num_tokens()
                
        self.dataset[f"{self.prefix}result"] = self.dataset.apply(lambda x: None, axis=1)
        self.dataset[f"{self.prefix}full_answer"] = self.dataset.apply(lambda x: None, axis=1)
        self.dataset[f"{self.prefix}is_skipped"] = self.dataset.apply(lambda x: None, axis=1)
        self.dataset[f"{self.prefix}is_prompt_too_long"] = self.dataset.apply(lambda x: None, axis=1)
        
        self._verify_after_processing_dataset_format(self.dataset)
        
        self.data_prepared = True
        
        return self.dataset

    def _prepare_num_tokens(self):
        return self.dataset.apply(lambda x: len(self.provider.get_tokens(x[f'{self.prefix}prompt'])), axis=1)
    
    def _prepare_prompts(self):
        return self.dataset.apply(lambda x: replace_from_dict(
                text=self.template, 
                pattern={
                    "system_prompt": self.system_prompt,
                    "data": f"QUERY=\"{x[self.query_column]}\" DESCRIPTION=\"{x['description']}\" CONTEXT=\"{x['context']}\"",
                    "prompt": self.prompt,
                    "lead_answer_prompt": self.lead_answer_prompt
                    }),
            axis=1)
        

    def load_dataframe(self, dataset_path: str):
        self.dataset_path = dataset_path
        
        cleaned_queries = load_dataset(self.dataset_path)

        self.raw_dataset = pd.DataFrame(cleaned_queries)
        self._verify_base_dataset_format(self.raw_dataset)
        
        self.dataset = self.raw_dataset.copy()
        self.data_loaded = True
        return self.dataset

    def get_dataset(self) -> pd.DataFrame:
        if not self.data_prepared:
            self.prepare_dataset()
        return self.dataset
    
    def _verify_base_dataset_format(self, dataset: pd.DataFrame):
        columns = [self.query_column, "context", "description"]
        for col in columns:
            if col not in dataset.columns:
                raise ValueError(f"Dataset does not contain the column {col}.")
    
    def _verify_after_processing_dataset_format(self, dataset: pd.DataFrame):
        columns = [
            self.query_column,
            "context",
            "description",
            f"{self.prefix}prompt",
            f"{self.prefix}num_tokens",
            f"{self.prefix}result",
            f"{self.prefix}full_answer",
            f"{self.prefix}is_skipped",
            f"{self.prefix}is_prompt_too_long"
        ]
        for col in columns:
            if col not in dataset.columns:
                raise ValueError(f"Dataset does not contain the column {col}.")