from abc import ABC, abstractmethod
from .Pipeline import Pipeline
import pandas as pd

class PipelineFeeder(ABC):
    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline
        self.results = list()
        
    @abstractmethod
    def process(self, iterable) -> list:
        pass
    
class SimplePipelineFeeder(PipelineFeeder):
    def __init__(self, pipeline: Pipeline) -> None:
        super().__init__(pipeline)
        self.implemented_type = [list, pd.DataFrame]

    def process(self, iterable):
        self.verify_type_is_implemented(iterable)
        
        if isinstance(iterable, list):
            for item in iterable:
                context = dict()
                context["row"] = item
                result = self.pipeline.execute(context)
                self.results.append(result)
        elif isinstance(iterable, pd.DataFrame):
            for item in range(len(iterable)):
                context = dict()
                context["row"] = iterable.iloc[item].to_dict()
                result = self.pipeline.execute(context)
                self.results.append(result)
        return self.results

    def verify_type_is_implemented(self, iterable):
        type_is_implemented = False
        for _type in self.implemented_type:
            if isinstance(iterable, _type):
                type_is_implemented = True
                break
            
        if not type_is_implemented:
            raise NotImplementedError(f"This type of iterable ({type(iterable)}) is not implemented.")