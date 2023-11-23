import argparse
import unittest
from libsparqltotext import DataWorkflowController, SaveService, DataProcessor, ContinuousDataLoader, TargetedDataLoader
import pandas as pd

from libsparqltotext.AnswerProcessor import BaseAnswerProcessor
from libsparqltotext.Provider import BaseProvider

class MockupSaveService(SaveService):
    def __init__(self) -> None:
        pass
    
    def export_save(self, last_index_row_processed: int) -> None:
        pass

class MockupDataProcessor(DataProcessor):
    def __init__(self) -> None:
        self.test_case = 0
    
    def process_row_number(self, row_index: int):
        if self.test_case == 0:
            return (None, None, True, True)
        elif self.test_case == 1:
            return (["answer1", "answer2"], "The full answer", False, False)
        elif self.test_case == 2:
            return (None, None, True, False)
    
class DataWorkflowControllerTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
    
    def test_constructor_makes_correct_dataloader_continuous(self):
        saveService = MockupSaveService()
        saveService.is_resumed = False
        dataset = pd.DataFrame()
        dataProcessor = MockupDataProcessor()
        
        workflow = DataWorkflowController(saveService, dataProcessor, dataset, "continuous", 0, 0, [], False, False)
        
        self.assertIsInstance(workflow.dataloader, ContinuousDataLoader)
    
    def test_constructor_makes_correct_dataloader_targeted(self):
        saveService = MockupSaveService()
        saveService.is_resumed = False
        dataset = pd.DataFrame()
        dataProcessor = MockupDataProcessor()
        
        workflow = DataWorkflowController(saveService, dataProcessor, dataset, "targeted", 0, 0, [], False, False)
        
        self.assertIsInstance(workflow.dataloader, TargetedDataLoader)
        
    def test_constructor_makes_correct_dataloader_skipped(self):
        saveService = MockupSaveService()
        saveService.is_resumed = False
        dataset = pd.DataFrame()
        dataset["is_skipped"] = pd.Series()
        dataProcessor = MockupDataProcessor()
        
        workflow = DataWorkflowController(saveService, dataProcessor, dataset, "skipped", 0, 0, [], False, False)
        
        self.assertIsInstance(workflow.dataloader, TargetedDataLoader)
    
    def test_constructor_makes_correct_dataloader_generation_type_doesnt_exist(self):
        saveService = MockupSaveService()
        saveService.is_resumed = False
        dataset = pd.DataFrame()
        dataset["is_skipped"] = pd.Series()
        dataProcessor = MockupDataProcessor()
        
        with self.assertRaises(ValueError):
            DataWorkflowController(saveService, dataProcessor, dataset, "inventing a generation type right now", 0, 0, [], False, False)
    
    def test_constructor_starting_row_new_generation_no_offset_all_dataset(self):
        saveService = MockupSaveService()
        saveService.is_resumed = False
        dataset = pd.DataFrame()
        dataset["value"] = pd.Series(data=[1, 2, 3])
        dataProcessor = MockupDataProcessor()
        
        workflow = DataWorkflowController(saveService, dataProcessor, dataset, "continuous", 0, 0, [], False, False)
        
        self.assertEqual(workflow.starting_row, 0)
        self.assertEqual(workflow.last_row_index, 3)
    
    def test_constructor_starting_row_new_generation_no_offset_all_dataset(self):
        saveService = MockupSaveService()
        saveService.is_resumed = False
        dataset = pd.DataFrame()
        dataset["value"] = pd.Series(data=[1, 2, 3])
        dataProcessor = MockupDataProcessor()
        
        workflow = DataWorkflowController(saveService, dataProcessor, dataset, "continuous", 0, -1, [], False, False)
        
        self.assertEqual(workflow.starting_row, 0)
        self.assertEqual(workflow.last_row_index, 3)
    
    def test_constructor_starting_row_new_generation_no_offset_1_row(self):
        saveService = MockupSaveService()
        saveService.is_resumed = False
        dataset = pd.DataFrame()
        dataset["value"] = pd.Series(data=[1, 2, 3])
        dataProcessor = MockupDataProcessor()
        
        workflow = DataWorkflowController(saveService, dataProcessor, dataset, "continuous", 0, 1, [], False, False)
        
        self.assertEqual(workflow.starting_row, 0)
        self.assertEqual(workflow.last_row_index, 1)
    
    def test_constructor_starting_row_new_generation_1_offset_all_dataset(self):
        saveService = MockupSaveService()
        saveService.is_resumed = False
        dataset = pd.DataFrame()
        dataset["value"] = pd.Series(data=[1, 2, 3])
        dataProcessor = MockupDataProcessor()
        
        workflow = DataWorkflowController(saveService, dataProcessor, dataset, "continuous", 1, 0, [], False, False)
        
        self.assertEqual(workflow.starting_row, 1)
        self.assertEqual(workflow.last_row_index, 3)
    
    def test_constructor_starting_row_new_generation_1_offset_all_dataset2(self):
        saveService = MockupSaveService()
        saveService.is_resumed = False
        dataset = pd.DataFrame()
        dataset["value"] = pd.Series(data=[1, 2, 3])
        dataProcessor = MockupDataProcessor()
        
        workflow = DataWorkflowController(saveService, dataProcessor, dataset, "continuous", 1, -1, [], False, False)
        
        self.assertEqual(workflow.starting_row, 1)
        self.assertEqual(workflow.last_row_index, 3)
        
    def test_constructor_starting_row_new_generation_1_offset_1_row(self):
        saveService = MockupSaveService()
        saveService.is_resumed = False
        dataset = pd.DataFrame()
        dataset["value"] = pd.Series(data=[1, 2, 3])
        dataProcessor = MockupDataProcessor()
        
        workflow = DataWorkflowController(saveService, dataProcessor, dataset, "continuous", 1, 1, [], False, False)
        
        self.assertEqual(workflow.starting_row, 1)
        self.assertEqual(workflow.last_row_index, 2)
        
    def test_constructor_starting_row_recovered_generation_no_offset_all_dataset(self):
        saveService = MockupSaveService()
        saveService.is_resumed = True
        saveService.last_index_row_processed = 0
        dataset = pd.DataFrame()
        dataset["value"] = pd.Series(data=[1, 2, 3])
        dataProcessor = MockupDataProcessor()
        
        workflow = DataWorkflowController(saveService, dataProcessor, dataset, "continuous", 0, 0, [], False, False)
        
        self.assertEqual(workflow.starting_row, 1)
        self.assertEqual(workflow.last_row_index, 3)
        
    def test_constructor_starting_row_recovered_generation_1_offset_all_dataset(self):
        saveService = MockupSaveService()
        saveService.is_resumed = True
        saveService.last_index_row_processed = 1
        dataset = pd.DataFrame()
        dataset["value"] = pd.Series(data=[1, 2, 3])
        dataProcessor = MockupDataProcessor()
        
        workflow = DataWorkflowController(saveService, dataProcessor, dataset, "continuous", 1, 0, [], False, False)
        
        self.assertEqual(workflow.starting_row, 2)
        self.assertEqual(workflow.last_row_index, 3)
        
    def test_constructor_starting_row_recovered_generation_1_offset_1_row(self):
        saveService = MockupSaveService()
        saveService.is_resumed = True
        saveService.last_index_row_processed = 1
        dataset = pd.DataFrame()
        dataset["value"] = pd.Series(data=[1, 2, 3])
        dataProcessor = MockupDataProcessor()
        
        workflow = DataWorkflowController(saveService, dataProcessor, dataset, "continuous", 1, 1, [], False, False)
        
        self.assertEqual(workflow.starting_row, 2)
        self.assertEqual(workflow.last_row_index, 2)
        
    def test_generate_valid(self):
        saveService = MockupSaveService()
        saveService.is_resumed = False
        dataset = pd.DataFrame()
        dataset["result"] = pd.Series(data=[None])
        dataset["full_answer"] = pd.Series(data=[None])
        dataset["is_skipped"] = pd.Series(data=[None])
        dataset["is_prompt_too_long"] = pd.Series(data=[None])
        dataProcessor = MockupDataProcessor()
        dataProcessor.test_case = 1
        # (["answer1", "answer2"], "The full answer", False, False)
        
        workflow = DataWorkflowController(saveService, dataProcessor, dataset, "continuous", 0, 1, [], False, False)
        workflow.generate()
        
        self.assertEqual(dataset["result"].iat[0], ["answer1", "answer2"])
        self.assertEqual(dataset["full_answer"].iat[0], "The full answer")
        self.assertEqual(dataset["is_skipped"].iat[0], False)
        self.assertEqual(dataset["is_prompt_too_long"].iat[0], False)
        
    def test_generate_skipped_too_long(self):
        saveService = MockupSaveService()
        saveService.is_resumed = False
        dataset = pd.DataFrame()
        dataset["result"] = pd.Series(data=[None])
        dataset["full_answer"] = pd.Series(data=[None])
        dataset["is_skipped"] = pd.Series(data=[None])
        dataset["is_prompt_too_long"] = pd.Series(data=[None])
        dataProcessor = MockupDataProcessor()
        dataProcessor.test_case = 0
        # (None, None, True, True)
        
        workflow = DataWorkflowController(saveService, dataProcessor, dataset, "continuous", 0, 1, [], False, False)
        workflow.generate()
        
        self.assertEqual(dataset["result"].iat[0], None)
        self.assertEqual(dataset["full_answer"].iat[0], None)
        self.assertEqual(dataset["is_skipped"].iat[0], True)
        self.assertEqual(dataset["is_prompt_too_long"].iat[0], True)
        
    def test_generate_too_many_tries(self):
        saveService = MockupSaveService()
        saveService.is_resumed = False
        dataset = pd.DataFrame()
        dataset["result"] = pd.Series(data=[None])
        dataset["full_answer"] = pd.Series(data=[None])
        dataset["is_skipped"] = pd.Series(data=[None])
        dataset["is_prompt_too_long"] = pd.Series(data=[None])
        dataProcessor = MockupDataProcessor()
        dataProcessor.test_case = 2
        # (None, None, True, False)
        
        workflow = DataWorkflowController(saveService, dataProcessor, dataset, "continuous", 0, 1, [], False, False)
        workflow.generate()
        
        self.assertEqual(dataset["result"].iat[0], None)
        self.assertEqual(dataset["full_answer"].iat[0], None)
        self.assertEqual(dataset["is_skipped"].iat[0], True)
        self.assertEqual(dataset["is_prompt_too_long"].iat[0], False)
        
        
        
    