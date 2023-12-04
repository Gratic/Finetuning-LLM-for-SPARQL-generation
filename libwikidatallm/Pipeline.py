from abc import ABC, abstractmethod

class PipelineStep(ABC):
    @abstractmethod
    def execute(self, context: dict):
        pass

class Pipeline(ABC):
    @abstractmethod
    def add_step(self, step: PipelineStep):
        pass
    
    @abstractmethod
    def execute(self, context: dict) -> dict:
        pass
    
class OrderedPipeline(Pipeline):
    def __init__(self) -> None:
        self.steps: list[PipelineStep] = list()
    
    def add_step(self, step: PipelineStep):
        if not isinstance(step, PipelineStep):
            raise TypeError("A step must be a child of PipelineStep class.")
        
        self.steps.append(step)
    
    def execute(self, context: dict) -> dict:
        for step in self.steps:
            step.execute(context)
            
        return context