import unittest
from libwikidatallm import PipelineStep, SimplePipelineFeeder, Pipeline
import pandas as pd

class PipelineDummy(Pipeline):
    def add_step(self, step: PipelineStep):
        raise NotImplementedError()
    
    def execute(self, context: dict) -> dict:
        return context

class SimplePipelineFeederTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        
    def test_initialisation_test(self):
        pipeline = PipelineDummy()
        spf = SimplePipelineFeeder(pipeline)
        
        self.assertEqual(spf.results, [])
        
    def test_feeding_a_list(self):
        dataset = [0, 1, 2]
        pipeline = PipelineDummy()
        spf = SimplePipelineFeeder(pipeline)
        
        results = spf.process(dataset)
        self.assertIs(spf.results, results)
        self.assertEqual(len(results), 3)
        self.assertEqual(results, [{"row": 0}, {"row": 1}, {"row": 2}])
        
    def test_cant_feed_a_dict(self):
        dataset = {"row0": 0, "row1": 1, "row2": 2}
        pipeline = PipelineDummy()
        spf = SimplePipelineFeeder(pipeline)
        
        with self.assertRaises(NotImplementedError):
            spf.process(dataset)
    
    def test_feeding_a_DataFrame(self):
        dataset = pd.DataFrame()
        dataset["value"] = pd.Series(data=[0,1,2])
        dataset["name"] = pd.Series(data=["row0", "row1", "row2"])
        pipeline = PipelineDummy()
        spf = SimplePipelineFeeder(pipeline)
        
        results = spf.process(dataset)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(results, [{"row": {"value": 0, "name": "row0"}}, {"row": {"value": 1, "name": "row1"}}, {"row": {"value": 2, "name": "row2"}}])