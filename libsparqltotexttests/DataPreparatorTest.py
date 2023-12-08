from typing import List
import unittest
from libsparqltotext import DataPreparator, BaseProvider
import pandas as pd
import os

class MockupProvider(BaseProvider):
    def query(self, parameters: dict[str, str | int | float]) -> bool:
        pass
    
    def get_tokens(self, parameters: dict[str, str | int | float]) -> List[int]:
        return [1 for word in parameters['content']]
    
class DataPreparatorTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
    
    def setUp(self):
        self.template = "<s>[INST] [system_prompt] [data] [prompt] [/INST] [lead_answer_prompt]"
    
    @classmethod
    def setUpClass(cls) -> None:
        # preparing mockup dataset files
        mock_empty = pd.DataFrame()
        with open(f"mock_empty.json", 'w') as f:
            f.write(mock_empty.to_json())
        
        no_good_cols = pd.DataFrame()
        no_good_cols["value"] = pd.Series()
        no_good_cols["wow"] = pd.Series()
        with open(f"no_good_cols.json", 'w') as f:
            f.write(no_good_cols.to_json())
        
        missing_cols = pd.DataFrame()
        missing_cols["query"] = pd.Series()
        missing_cols["description"] = pd.Series()
        with open(f"missing_cols.json", 'w') as f:
            f.write(missing_cols.to_json())
        
        missing_cols_2 = pd.DataFrame()
        missing_cols_2["description"] = pd.Series()
        missing_cols_2["context"] = pd.Series()
        with open(f"missing_cols_2.json", 'w') as f:
            f.write(missing_cols_2.to_json())
            
        missing_cols_3 = pd.DataFrame()
        missing_cols_3["query"] = pd.Series()
        missing_cols_3["context"] = pd.Series()
        with open(f"missing_cols_3.json", 'w') as f:
            f.write(missing_cols_3.to_json())
        
        good_dataset = pd.DataFrame()
        good_dataset["query"] = pd.Series()
        good_dataset["context"] = pd.Series()
        good_dataset["description"] = pd.Series()
        with open(f"good_dataset.json", 'w') as f:
            f.write(good_dataset.to_json())
        
    @classmethod
    def tearDownClass(cls) -> None:
        # removing mockup dataset
        os.remove("mock_empty.json")
        os.remove("no_good_cols.json")
        os.remove("missing_cols.json")
        os.remove("missing_cols_2.json")
        os.remove("missing_cols_3.json")
        os.remove("good_dataset.json")
            
    def test_verify_base_dataset_format_empty(self):
        dataset = pd.DataFrame()
        
        with self.assertRaises(ValueError):
            DataPreparator._verify_base_dataset_format(dataset)
    
    def test_verify_base_dataset_format_empty2(self):
        dataset = pd.DataFrame()
        dataset["nope"] = pd.Series()
        
        with self.assertRaises(ValueError):
            DataPreparator._verify_base_dataset_format(dataset)
    
    def test_verify_base_dataset_format_good(self):
        dataset = pd.DataFrame()
        dataset["query"] = pd.Series()
        dataset["description"] = pd.Series()
        dataset["context"] = pd.Series()
        
        DataPreparator._verify_base_dataset_format(dataset)
        
    def test_verify_after_processing_dataset_format_empty(self):
        dataset = pd.DataFrame()
        
        with self.assertRaises(ValueError):
            DataPreparator._verify_after_processing_dataset_format(dataset)
            
    def test_verify_after_processing_dataset_format_empty2(self):
        dataset = pd.DataFrame()
        dataset["nope"] = pd.Series()
        
        with self.assertRaises(ValueError):
            DataPreparator._verify_after_processing_dataset_format(dataset)
    
    def test_verify_after_processing_dataset_format_not_all_columns(self):
        dataset = pd.DataFrame()
        dataset["query"] = pd.Series()
        dataset["description"] = pd.Series()
        dataset["context"] = pd.Series()
        
        with self.assertRaises(ValueError):
            DataPreparator._verify_after_processing_dataset_format(dataset)
    
    def test_verify_after_processing_dataset_format_good(self):
        dataset = pd.DataFrame()
        dataset["query"] = pd.Series()
        dataset["description"] = pd.Series()
        dataset["context"] = pd.Series()
        dataset["prompt"] = pd.Series()
        dataset["num_tokens"] = pd.Series()
        dataset["result"] = pd.Series()
        dataset["full_answer"] = pd.Series()
        dataset["is_skipped"] = pd.Series()
        dataset["is_prompt_too_long"] = pd.Series()
        
        DataPreparator._verify_after_processing_dataset_format(dataset)
    
    def test_prepare_dataset_pp_yes_valid(self):
        dataset = pd.DataFrame()
        dataset["query"] = pd.Series(data=["my query"])
        dataset["description"] = pd.Series(data=["my description"])
        dataset["context"] = pd.Series(data=["my context"])
        
        dataPreparator = DataPreparator(MockupProvider(), self.template, "system prompt", "prompt", "lead", "yes")
        dataPreparator.data_loaded = True
        dataPreparator.dataset = dataset
        
        self.assertFalse(dataPreparator.data_prepared)
        self.assertTrue(isinstance(dataPreparator.prepare_dataset(), pd.DataFrame))
        self.assertTrue(dataPreparator.data_prepared)
        
        colnames = ["query", "context", "description", "prompt", "result", "full_answer", "is_skipped", "is_prompt_too_long"]
        for col in colnames:
            if col not in dataPreparator.get_dataset().columns:
                self.fail(f"{col} is not in the prepared dataset.")
        
        self.assertEqual(dataPreparator.get_dataset()['prompt'].iat[0], "<s>[INST] system prompt QUERY=\"my query\" DESCRIPTION=\"my description\" CONTEXT=\"my context\" prompt [/INST] lead")
        
        self.assertEqual(dataPreparator.get_dataset()['result'].iat[0], None)
        self.assertEqual(dataPreparator.get_dataset()['full_answer'].iat[0], None)
        self.assertEqual(dataPreparator.get_dataset()['is_skipped'].iat[0], None)
        self.assertEqual(dataPreparator.get_dataset()['is_prompt_too_long'].iat[0], None)
        
    def test_prepare_dataset_pp_auto_valid(self):
        dataset = pd.DataFrame()
        dataset["query"] = pd.Series(data=["my query"])
        dataset["description"] = pd.Series(data=["my description"])
        dataset["context"] = pd.Series(data=["my context"])
        
        dataPreparator = DataPreparator(MockupProvider(), self.template, "system prompt", "prompt", "lead", "auto")
        dataPreparator.data_loaded = True
        dataPreparator.dataset = dataset
        
        self.assertFalse(dataPreparator.data_prepared)
        self.assertTrue(isinstance(dataPreparator.prepare_dataset(), pd.DataFrame))
        self.assertTrue(dataPreparator.data_prepared)
        
        colnames = ["query", "context", "description", "prompt", "result", "full_answer", "is_skipped", "is_prompt_too_long"]
        for col in colnames:
            if col not in dataPreparator.get_dataset().columns:
                self.fail(f"{col} is not in the prepared dataset.")
        
        self.assertEqual(dataPreparator.get_dataset()['prompt'].iat[0], "<s>[INST] system prompt QUERY=\"my query\" DESCRIPTION=\"my description\" CONTEXT=\"my context\" prompt [/INST] lead")
        
        self.assertEqual(dataPreparator.get_dataset()['result'].iat[0], None)
        self.assertEqual(dataPreparator.get_dataset()['full_answer'].iat[0], None)
        self.assertEqual(dataPreparator.get_dataset()['is_skipped'].iat[0], None)
        self.assertEqual(dataPreparator.get_dataset()['is_prompt_too_long'].iat[0], None)
        
    def test_prepare_dataset_pp_auto_with_prompt_valid(self):
        dataset = pd.DataFrame()
        dataset["query"] = pd.Series(data=["my query"])
        dataset["description"] = pd.Series(data=["my description"])
        dataset["context"] = pd.Series(data=["my context"])
        dataset["prompt"] = pd.Series(data=["my supreme prompt"])
        
        dataPreparator = DataPreparator(MockupProvider(), self.template, "system prompt", "prompt", "lead", "auto")
        dataPreparator.data_loaded = True
        dataPreparator.dataset = dataset
        
        self.assertFalse(dataPreparator.data_prepared)
        self.assertTrue(isinstance(dataPreparator.prepare_dataset(), pd.DataFrame))
        self.assertTrue(dataPreparator.data_prepared)
        
        colnames = ["query", "context", "description", "prompt", "result", "full_answer", "is_skipped", "is_prompt_too_long"]
        for col in colnames:
            if col not in dataPreparator.get_dataset().columns:
                self.fail(f"{col} is not in the prepared dataset.")
        
        self.assertEqual(dataPreparator.get_dataset()['prompt'].iat[0], "my supreme prompt")
        
        self.assertEqual(dataPreparator.get_dataset()['result'].iat[0], None)
        self.assertEqual(dataPreparator.get_dataset()['full_answer'].iat[0], None)
        self.assertEqual(dataPreparator.get_dataset()['is_skipped'].iat[0], None)
        self.assertEqual(dataPreparator.get_dataset()['is_prompt_too_long'].iat[0], None)
    
    def test_prepare_dataset_pp_no_valid(self):
        dataset = pd.DataFrame()
        dataset["query"] = pd.Series(data=["my query"])
        dataset["description"] = pd.Series(data=["my description"])
        dataset["context"] = pd.Series(data=["my context"])
        dataset["prompt"] = pd.Series(data=["my turbo prompt"])
        dataset["num_tokens"] = pd.Series(data=[3])
        
        dataPreparator = DataPreparator(MockupProvider(), self.template, "system prompt", "prompt", "lead", "no")
        dataPreparator.data_loaded = True
        dataPreparator.dataset = dataset
        
        self.assertFalse(dataPreparator.data_prepared)
        self.assertTrue(isinstance(dataPreparator.prepare_dataset(), pd.DataFrame))
        self.assertTrue(dataPreparator.data_prepared)
        
        colnames = ["query", "context", "description", "prompt", "result", "full_answer", "is_skipped", "is_prompt_too_long"]
        for col in colnames:
            if col not in dataPreparator.get_dataset().columns:
                self.fail(f"{col} is not in the prepared dataset.")
        
        self.assertEqual(dataPreparator.get_dataset()['prompt'].iat[0], "my turbo prompt")
        
        self.assertEqual(dataPreparator.get_dataset()['result'].iat[0], None)
        self.assertEqual(dataPreparator.get_dataset()['full_answer'].iat[0], None)
        self.assertEqual(dataPreparator.get_dataset()['is_skipped'].iat[0], None)
        self.assertEqual(dataPreparator.get_dataset()['is_prompt_too_long'].iat[0], None)
        
    def test_prepare_dataset_pp_no_but_no_prompt_column(self):
        dataset = pd.DataFrame()
        dataset["query"] = pd.Series(data=["my query"])
        dataset["description"] = pd.Series(data=["my description"])
        dataset["context"] = pd.Series(data=["my context"])
        
        dataPreparator = DataPreparator(MockupProvider(), self.template, "system prompt", "prompt", "lead", "no")
        dataPreparator.data_loaded = True
        dataPreparator.dataset = dataset
        
        self.assertFalse(dataPreparator.data_prepared)
        with self.assertRaises(ValueError):
            dataPreparator.prepare_dataset()
    
    def test_load_dataframe_empty(self):
        dataPreparator = DataPreparator(MockupProvider(), "", "", "", "", "")
        self.assertEqual(dataPreparator.data_loaded, False)
        self.assertEqual(dataPreparator.raw_dataset, None)
        self.assertEqual(dataPreparator.dataset_path, None)
        
        with self.assertRaises(ValueError):
            dataPreparator.load_dataframe("mock_empty.json")
    
    def test_load_dataframe_no_good_cols(self):
        dataPreparator = DataPreparator(MockupProvider(), "", "", "", "", "")
        self.assertEqual(dataPreparator.data_loaded, False)
        self.assertEqual(dataPreparator.raw_dataset, None)
        self.assertEqual(dataPreparator.dataset_path, None)
        
        with self.assertRaises(ValueError):
            dataPreparator.load_dataframe("no_good_cols.json")
    
    def test_load_dataframe_missing_cols(self):
        dataPreparator = DataPreparator(MockupProvider(), "", "", "", "", "")
        self.assertEqual(dataPreparator.data_loaded, False)
        self.assertEqual(dataPreparator.raw_dataset, None)
        self.assertEqual(dataPreparator.dataset_path, None)
        
        with self.assertRaises(ValueError):
            dataPreparator.load_dataframe("missing_cols.json")
    
    def test_load_dataframe_missing_cols_2(self):
        dataPreparator = DataPreparator(MockupProvider(), "", "", "", "", "")
        self.assertEqual(dataPreparator.data_loaded, False)
        self.assertEqual(dataPreparator.raw_dataset, None)
        self.assertEqual(dataPreparator.dataset_path, None)
        
        with self.assertRaises(ValueError):
            dataPreparator.load_dataframe("missing_cols_2.json")
    
    def test_load_dataframe_missing_cols_3(self):
        dataPreparator = DataPreparator(MockupProvider(), "", "", "", "", "")
        self.assertEqual(dataPreparator.data_loaded, False)
        self.assertEqual(dataPreparator.raw_dataset, None)
        self.assertEqual(dataPreparator.dataset_path, None)
        
        with self.assertRaises(ValueError):
            dataPreparator.load_dataframe("missing_cols_3.json")
    
    def test_load_dataframe_good_dataset(self):
        dataPreparator = DataPreparator(MockupProvider(), "", "", "", "", "")
        self.assertEqual(dataPreparator.data_loaded, False)
        self.assertEqual(dataPreparator.raw_dataset, None)
        self.assertEqual(dataPreparator.dataset_path, None)
        
        self.assertTrue(isinstance(dataPreparator.load_dataframe("good_dataset.json"), pd.DataFrame))
        
        self.assertTrue(dataPreparator.data_loaded)
        self.assertTrue(dataPreparator.dataset_path, "good_dataset.json")
        self.assertTrue(isinstance(dataPreparator.raw_dataset, pd.DataFrame))
        self.assertTrue(isinstance(dataPreparator.dataset, pd.DataFrame))
        self.assertIsNot(dataPreparator.raw_dataset, dataPreparator.dataset)
            
    