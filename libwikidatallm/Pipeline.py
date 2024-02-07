from abc import ABC, abstractmethod

class NoSparqlMatchError(Exception):
    def __init__(self, msg, sparql, *args: object) -> None:
        super().__init__(*args)
        self.msg = msg
        self.sparql = sparql

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
        try:
            context['last_executed_step'] = ""
            for step in self.steps:
                context['to_be_executed_step'] = type(step).__name__
                step.execute(context)
                context['last_executed_step'] = context['to_be_executed_step']
                
            context['status'] = ""
            context['has_error'] = False
            context['to_be_executed_step'] = ""
        except NoSparqlMatchError as err:
            context['status'] = f"Unexpected {err.msg=}, {type(err)=}: {err.sparql=}"
            context['has_error'] = True
        except Exception as err:
            context['status'] = f"Unexpected {err=}, {type(err)=}"
            context['has_error'] = True
            
        return context