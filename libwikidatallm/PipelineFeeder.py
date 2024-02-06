from abc import ABC, abstractmethod
from .Pipeline import Pipeline
import pandas as pd
from typing import Union, List, Any
from tqdm.auto import tqdm

class PipelineFeeder(ABC):
    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline
        self.results: List[dict] = list()
        
    @abstractmethod
    def process(self, iterable) -> List[dict]:
        pass
    
class SimplePipelineFeeder(PipelineFeeder):
    def __init__(self, pipeline: Pipeline, use_tqdm=False) -> None:
        super().__init__(pipeline)
        self.implemented_type = [list, pd.DataFrame, pd.Series]
        self.use_tqdm = use_tqdm

    def process(self, iterable: Union[List[Any], pd.DataFrame]):
        self.ensure_supported_iterable_type(iterable)
        
        if isinstance(iterable, list):
            for row in (tqdm(iterable) if self.use_tqdm else iterable):
                self.process_row(row)
        elif isinstance(iterable, pd.DataFrame):
            for _, row in (tqdm(iterable.iterrows()) if self.use_tqdm else iterable.iterrows()):
                self.process_row(row.to_dict())
        elif isinstance(iterable, pd.Series):
            for _, row in (tqdm(iterable.items()) if self.use_tqdm else iterable.items()):
                self.process_row(row)
        return self.results

    def process_row(self, item: Any):
        context = dict()
        context["row"] = item
        result = self.pipeline.execute(context)
        self.results.append(result)

    def ensure_supported_iterable_type(self, iterable: Any):            
        if not any([isinstance(iterable, _type) for _type in self.implemented_type]):
            supported_type = [implemented.__name__ for implemented in self.implemented_type]
            raise NotImplementedError(f"This type ({type(iterable).__name__}) is not implemented. Supported types are: {', '.join(supported_type)}.")