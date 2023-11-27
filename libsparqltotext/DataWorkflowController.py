from tqdm import tqdm
from typing import List
from .Provider import BaseProvider
from .SaveService import SaveService
import pandas as pd
from .DataLoader import ContinuousDataLoader, TargetedDataLoader
from .DataProcessor import DataProcessor    

class DataWorkflowController():
    def __init__(self, saveService: SaveService, dataProcessor: DataProcessor, dataset: pd.DataFrame, generation_type: str, offset: int, number_of_rows: int, targets: List[int], verbose: bool) -> None:
        self.saveService: SaveService = saveService
        self.dataProcessor: DataProcessor = dataProcessor
        self.dataset: pd.DataFrame = dataset
        
        self.generation_type: str = generation_type
        self.offset: int = offset
        self.number_of_rows: int = number_of_rows
        self.targets: List[int] = targets
        self.verbose: bool = verbose
        
        self.dataloader = None
        if self.generation_type == "continuous":
            self.starting_row: int = saveService.last_index_row_processed + 1 if saveService.is_resumed_generation() else self.offset
            self.last_row_index: int = len(self.dataset) if self.number_of_rows <= 0 else self.offset + self.number_of_rows
            self.dataloader = ContinuousDataLoader(self.dataset, self.starting_row, self.last_row_index)
        elif self.generation_type == "targeted" or self.generation_type == "skipped":
            self.dataloader = TargetedDataLoader(self.dataset, self.targets)
        else:
            raise ValueError(f"Generation type given {self.generation_type} does not exists. Choices are \"continuous\", \"targeted\", \"skipped\".")
    
    def generate(self):
        if self.verbose:
            print("Generating reverse prompts... ")
            
        for row in tqdm(self.dataloader):
            row_index = int(row.name)
            (results, full_answer, skipped, context_too_long) = self.dataProcessor.process_row_number(row_index)
            
            if skipped and context_too_long:
                print(f"Prompt has too much token at row: {row_index}")
            elif skipped:
                print(f"No results found for: {row_index}")
                
            self.dataset['result'].iat[row_index] = results
            self.dataset['full_answer'].iat[row_index] = full_answer
            self.dataset["is_skipped"].iat[row_index] = skipped
            self.dataset["is_prompt_too_long"].iat[row_index] = context_too_long
            
            self.saveService.export_save(row_index)