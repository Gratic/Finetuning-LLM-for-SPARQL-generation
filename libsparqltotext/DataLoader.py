import pandas as pd
from typing import List

class BaseDataLoader():
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