from tqdm import tqdm
from typing import List
from .Provider import BaseProvider
from .SaveService import SaveService
from .RegexService import RegexService
import pandas as pd
from abc import ABC, abstractmethod

RETRY_IF_ANSWER_CONTAINS = ["SELECT", "GROUP"]

class DataProcessor():
    def __init__(self, provider: BaseProvider, regexService: RegexService, dataset: pd.DataFrame, retry_attempts: int, context_length_limit: int, prediction_size: int, temperature: float, print_answers: bool, print_results: bool) -> None:
        self.provider: BaseProvider = provider
        self.regexService: RegexService = regexService
        self.dataset: pd.DataFrame = dataset
        self.retry_attempts: int = retry_attempts
        self.context_length_limit: int = context_length_limit
        self.prediction_size: int = prediction_size
        self.temperature: float = temperature
        self.print_answers: bool = print_answers
        self.print_results: bool = print_results
        
        self.prompts: pd.Series = dataset['prompt']
        self.num_tokens: pd.Series = dataset['num_tokens']
    
    def process_row(self, row_index: int):
        '''Returns (results, full answer, skipped, context length too long)'''
        num_token = self.num_tokens.iat[row_index]
        
        if num_token > self.context_length_limit:
            return (None, None, True, True)
        
        data_json = self.prepare_request_payload(row_index)
        
        number_of_try_left = self.retry_attempts
        while number_of_try_left != 0:    
            self.provider.query(data_json)
            
            if self.print_answers:
                print(self.provider.get_answer())
            
            results = self.regexService.extract_prompts(self.provider.get_answer())
            
            if self.print_results:
                print(results)
            
            if not self.are_results_acceptable(results, RETRY_IF_ANSWER_CONTAINS):
                number_of_try_left -= 1
                continue
            
            return (results, self.provider.get_full_answer(), False, False)

        return (None, None, True, False)
            
    def prepare_request_payload(self, row_index: int) -> dict[str, str | int | float]:
        payload: dict[str,  str | int | float] = dict()
        payload["prompt"] = self.prompts.iat[row_index]
        payload["n_predict"] = self.prediction_size
        payload["temperature"] = self.temperature

        return payload

    def are_results_acceptable(results: List[str], banned_words: List[str]) -> bool:
        is_good_quality = True
        
        if len(results) == 0:
            return False
        
        for result in results:
            for word in banned_words:
                if word in result:
                    is_good_quality = False
        return is_good_quality


class BaseDataLoader(ABC):
    def __init__(self, dataset: pd.DataFrame) -> None:
        self.dataset: pd.DataFrame = dataset
        self.current_index = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_index < len(self.dataset):
            result = self.dataset.at[self.current_index]
            self.current_index += 1
            return result
        raise StopIteration

class ContinuousDataLoader(BaseDataLoader):
    def __init__(self, dataset: pd.DataFrame, starting_row: int, last_row_index: int) -> None:
        super().__init__(dataset)
        self.current_index = starting_row
        self.last_row_index = last_row_index
    
    def __next__(self):
        if self.current_index < len(self.dataset) and self.current_index < self.last_row_index:
            result = self.dataset.at[self.current_index]
            self.current_index += 1
            return result
        raise StopIteration

class TargetedDataLoader(BaseDataLoader):
    def __init__(self, dataset: pd.DataFrame, targets: List[int]) -> None:
        super().__init__(dataset)
        self.targets = targets
    
    def __next__(self):
        if self.current_index < len(self.targets):
            result = self.dataset.at[self.targets[self.current_index]]
            self.current_index += 1
            return result
        raise StopIteration
    
class DataWorkflowController():
    def __init__(self, provider: BaseProvider, saveService: SaveService, dataProcessor: DataProcessor, dataset: pd.DataFrame, generation_type: str, offset: int, number_of_rows: int, targets: List[int], verbose: bool, quiet: bool) -> None:
        self.provider: BaseProvider = provider
        self.saveService: SaveService = saveService
        self.dataProcessor: DataProcessor = dataProcessor
        self.dataset: pd.DataFrame = dataset
        
        self.generation_type: str = generation_type
        self.offset: int = offset
        self.number_of_rows: int = number_of_rows
        self.targets: List[int] = targets
        self.verbose: bool = verbose
        self.quiet: bool = quiet
        
        self.starting_row: int = saveService.last_index_row_processed + 1 if saveService.is_resumed_generation() else self.offset
        self.last_row_index: int = len(self.prompts) if self.number_of_rows <= 0 else self.offset + self.number_of_rows
    
    def generate(self):
        dataloader = None
        if self.generation_type == "continuous":
            dataloader = ContinuousDataLoader(self.dataset, self.starting_row, self.last_row_index)
        elif self.generation_type == "targeted":
            dataloader = TargetedDataLoader(self.dataset, self.targets)
        elif self.generation_type == "skipped":
            dataloader = TargetedDataLoader(self.dataset, list(self.dataset.loc[self.dataset["is_skipped"] == True].index))
        
        if self.verbose:
            print("Generating reverse prompts... ")
            
        for row_index in tqdm(dataloader):
            
            (results, full_answer, skipped, context_too_long) = self.dataProcessor.process_row(row_index)
            
            if skipped and context_too_long:
                print(f"Prompt has too much token at row: {self.dataset.iloc[row_index].name}")
            elif skipped:
                print(f"No results found for: {self.dataset.iloc[row_index].name}")
                
            self.dataset['result'].iat[row_index] = results
            self.dataset['full_answer'].iat[row_index] = full_answer
            self.dataset["is_skipped"].iat[row_index] = skipped
            self.dataset["is_prompt_too_long"].iat[row_index] = context_too_long
            
            self.saveService.export_save(row_index)