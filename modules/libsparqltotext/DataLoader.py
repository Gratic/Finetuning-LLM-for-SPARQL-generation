import pandas as pd
from typing import List

class BaseDataLoader():
    def __init__(self, dataset: pd.DataFrame) -> None:
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError("Only pandas DataFrame are allowed.")
        
        self.dataset: pd.DataFrame = dataset
        self.current_index = 0

    def __iter__(self):
        return self
    
    def __next__(self) -> pd.Series:
        if self.current_index < len(self.dataset):
            result = self.dataset.iloc[self.current_index]
            self.current_index += 1
            return result
        raise StopIteration

class ContinuousDataLoader(BaseDataLoader):
    '''
    Load data continuously from [\"starting_row\":\"last_row_index\"[. \"last_row_indew\" is not included.
    starting_row=0 and last_row_index=len(dataset) === BaseDataLoader
    '''
    def __init__(self, dataset: pd.DataFrame, starting_row: int, last_row_index: int) -> None:
        super().__init__(dataset)
        
        if starting_row < 0:
            raise ValueError("starting_row must be positive or zero.")
                
        self.current_index = starting_row
        self.last_row_index = last_row_index
    
    def __next__(self):
        if self.current_index < len(self.dataset) and self.current_index < self.last_row_index:
            result = self.dataset.iloc[self.current_index]
            self.current_index += 1
            return result
        raise StopIteration

class TargetedDataLoader(BaseDataLoader):
    def __init__(self, dataset: pd.DataFrame, targets: List[int]) -> None:
        super().__init__(dataset)
        
        if not isinstance(targets, List) or (len(targets) != 0 and not isinstance(targets[0], int)):
            raise TypeError("targets must be a list on integer.")
        
        self.targets = targets
    
    def __next__(self):
        if self.current_index < len(self.targets):
            result = self.dataset.iloc[self.targets[self.current_index]]
            self.current_index += 1
            return result
        raise StopIteration