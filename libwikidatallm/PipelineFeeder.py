from abc import ABC, abstractmethod
from .Pipeline import Pipeline

class PipelineFeeder(ABC):
    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline
        self.results = list()
        
    @abstractmethod
    def process(self, iterable):
        pass
    
class SimplePipelineFeeder(PipelineFeeder):
    def __init__(self, pipeline: Pipeline) -> None:
        super().__init__(pipeline)

    def process(self, iterable):
        for item in iterable:
            context = dict()
            context["row"] = item
            result = self.pipeline.execute(context)
            self.results.append(result)