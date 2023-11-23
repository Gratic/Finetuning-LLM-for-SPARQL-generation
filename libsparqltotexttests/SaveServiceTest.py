import unittest
from libsparqltotext import SaveService
import argparse
import pandas as pd
import json
import os

class SaveServiceTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
    
    @classmethod
    def setUpClass(cls) -> None:
        namespace = SaveServiceTest._mockup_namespace("exist")
        
        dataset = pd.DataFrame()
        dataset['query'] = pd.Series(data=["A query"])
        dataset['description'] = pd.Series(data=["A description"])
        dataset['context'] = pd.Series(data=["A context"])
        dataset['prompt'] = pd.Series(data=["A prompt"])
        dataset['num_tokens'] = pd.Series(data=[150])
        dataset['result'] = pd.Series(data=[['answer 1', 'answer 2']])
        dataset['full_answer'] = pd.Series(data=["A full_answer"])
        dataset['is_skipped'] = pd.Series(data=[False])
        dataset['is_prompt_too_long'] = pd.Series(data=[False])
        
        checkpoint_dict = dict()
        checkpoint_dict['args'] = namespace.__dict__
        checkpoint_dict['dataset'] = dataset.to_json()
        checkpoint_dict['last_index_row_processed'] = 0
        checkpoint_dict_json = json.dumps(checkpoint_dict)
        
        with open("exist.chk", 'w') as f:
            f.write(checkpoint_dict_json)

    @staticmethod
    def _mockup_namespace(save_identifier: str):
        namespace = argparse.Namespace()
        namespace.__dict__['test_case'] = True
        namespace.__dict__['save_identifier'] = save_identifier
        namespace.__dict__['checkpoint_path'] = ""
        return namespace
    
    @classmethod
    def tearDownClass(cls) -> None:
        os.remove("exist.chk")
        
    def setUp(self) -> None:
        self.namespace_exist = self._mockup_namespace("exist")
        self.namespace_not_exist = self._mockup_namespace("not_exist")
                
    def test_load_save_file_exist(self):
        saveService = SaveService(self.namespace_exist)
        
        test_dataset = pd.DataFrame()
        test_dataset['query'] = pd.Series(data=["A query"])
        test_dataset['description'] = pd.Series(data=["A description"])
        test_dataset['context'] = pd.Series(data=["A context"])
        test_dataset['prompt'] = pd.Series(data=["A prompt"])
        test_dataset['num_tokens'] = pd.Series(data=[150])
        test_dataset['result'] = pd.Series(data=[['answer 1', 'answer 2']])
        test_dataset['full_answer'] = pd.Series(data=["A full_answer"])
        test_dataset['is_skipped'] = pd.Series(data=[False])
        test_dataset['is_prompt_too_long'] = pd.Series(data=[False])
        
        self.assertIsNone(saveService.dataset)
        self.assertEqual(saveService.last_index_row_processed, -1)
        self.assertEqual(saveService.filepath, "exist.chk")
        
        (args, dataset, last_index_row_processed) = saveService.load_save()
        
        self.assertIsInstance(args, argparse.Namespace)
        self.assertIsInstance(dataset, pd.DataFrame)
        self.assertEqual(dataset['query'].iat[0], test_dataset['query'].iat[0])
        self.assertEqual(dataset['description'].iat[0], test_dataset['description'].iat[0])
        self.assertEqual(dataset['context'].iat[0], test_dataset['context'].iat[0])
        self.assertEqual(dataset['prompt'].iat[0], test_dataset['prompt'].iat[0])
        self.assertEqual(dataset['num_tokens'].iat[0], test_dataset['num_tokens'].iat[0])
        self.assertEqual(dataset['result'].iat[0], test_dataset['result'].iat[0])
        self.assertEqual(dataset['full_answer'].iat[0], test_dataset['full_answer'].iat[0])
        self.assertEqual(dataset['is_skipped'].iat[0], test_dataset['is_skipped'].iat[0])
        self.assertEqual(dataset['is_prompt_too_long'].iat[0], test_dataset['is_prompt_too_long'].iat[0])
        self.assertEqual(len(dataset), len(test_dataset))
        self.assertEqual(last_index_row_processed, 0)
        self.assertTrue(saveService.is_resumed)
        self.assertTrue(saveService.is_resumed_generation())
        self.assertFalse(saveService.is_new_generation())
                
    def test_load_save_file_not_exist(self):
        saveService = SaveService(self.namespace_not_exist)
        
        self.assertIsNone(saveService.dataset)
        self.assertEqual(saveService.last_index_row_processed, -1)
        self.assertEqual(saveService.filepath, "not_exist.chk")
        
        (args, dataset, last_index_row_processed) = saveService.load_save()
        
        self.assertIsInstance(args, argparse.Namespace)
        self.assertIsNone(dataset)
        self.assertEqual(last_index_row_processed, -1)
        self.assertFalse(saveService.is_resumed)
        self.assertFalse(saveService.is_resumed_generation())
        self.assertTrue(saveService.is_new_generation())
    
    def test_export_save_valid(self):
        namespace = self._mockup_namespace("to_save")
        saveService = SaveService(namespace)
        
        dataset = pd.DataFrame()
        dataset['value'] = pd.Series(data=["My value"])
        
        saveService.dataset = dataset
        
        saveService.export_save(0)
        
        self.assertTrue(os.path.exists("to_save.chk"))
        
        (args, load_dataset, last_index_row_processed) = saveService.load_save()
        
        self.assertEqual(args, namespace)
        self.assertEqual(last_index_row_processed, 0)
        self.assertEqual(load_dataset['value'].iat[0], dataset['value'].iat[0])
        self.assertTrue(saveService.is_resumed)
        self.assertTrue(saveService.is_resumed_generation())
        self.assertFalse(saveService.is_new_generation())
        
        os.remove("to_save.chk")