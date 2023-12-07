import unittest
from libwikidatallm.Pipeline import OrderedPipeline, PipelineStep

class PipelineStepDummy(PipelineStep):
    def __init__(self, value = 0) -> None:
        self.value = value
        
    def execute(self, context: dict):
        if "test" in context.keys():
            context["test"].append(self.value)
        else:
            context["test"] = [self.value]

class PipelineStepRaisesErrorDummy(PipelineStep):
    def execute(self, context: dict):
        raise ValueError("Testing exception handling.")

class OrderedPipelineTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        
    def test_init_nothing(self):
        pipeline = OrderedPipeline()
        self.assertEqual(pipeline.steps, list())
    
    def test_append_pipeline_step(self):
        pipelineStep = PipelineStepDummy()
        pipeline = OrderedPipeline()
        pipeline.add_step(pipelineStep)
        
        self.assertEqual(pipeline.steps, [pipelineStep])
        self.assertEqual(len(pipeline.steps), 1)
    
    def test_append_not_pipeline_step(self):
        pipelineStep = list()
        pipeline = OrderedPipeline()
        
        with self.assertRaises(TypeError):
            pipeline.add_step(pipelineStep)
    
    def test_execution_empty(self):
        pipeline = OrderedPipeline()
        pipeline.execute({})
    
    def test_execution_one_step(self):
        pipelineStep = PipelineStepDummy(0)
        pipeline = OrderedPipeline()
        pipeline.add_step(pipelineStep)
        
        context = {}
        self.assertEqual(pipeline.execute(context), {"test": [0]})
    
    def test_execution_three_step(self):
        pipelineStep0 = PipelineStepDummy(0)
        pipelineStep1 = PipelineStepDummy(1)
        pipelineStep2 = PipelineStepDummy(2)
        pipeline = OrderedPipeline()
        pipeline.add_step(pipelineStep0)
        pipeline.add_step(pipelineStep1)
        pipeline.add_step(pipelineStep2)
        
        context = {}
        self.assertEqual(pipeline.execute(context), {"test": [0, 1 ,2]})
    
    def test_pipeline_step_raises_error(self):
        pipelineStep = PipelineStepRaisesErrorDummy()
        pipeline = OrderedPipeline()
        pipeline.add_step(pipelineStep)
        
        with self.assertRaises(ValueError):
            pipeline.execute({})