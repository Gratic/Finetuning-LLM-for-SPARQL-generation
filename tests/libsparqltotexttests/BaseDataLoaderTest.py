import unittest
from modules.libsparqltotext import BaseDataLoader
import pandas as pd

class BaseDataLoaderTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        
    def test_normal_case(self):
        value = [10, 20, 30, 40]
        valueB = [90, 80, 70, 60]
        
        dataset = pd.DataFrame()
        dataset["value"] = pd.Series(data=value)
        dataset["valueB"] = pd.Series(data=valueB)
        
        dataloader = BaseDataLoader(dataset)
        
        count = 0
        for x in dataloader:
            self.assertEqual(type(x), pd.Series, f"The dataloader doesn't return the pandas Series.")
            self.assertEqual(x["value"], value[count], f"Row order is not respected.")
            self.assertEqual(x["valueB"], valueB[count], f"Row order is not respected.")
            count += 1
        
        self.assertEqual(count, len(dataset), f"The number of row processed is not correct. It should have been {len(dataset)} but only {count} where iterated on.")
    
    def test_dataset_is_not_dataframe(self):
        dataset = [10, 20, 30, 40]
        
        with self.assertRaises(TypeError):
            BaseDataLoader(dataset)
    
    def test_dataset_is_empty(self):
        dataset = pd.DataFrame()
        
        dataloader = BaseDataLoader(dataset)
        
        for x in dataloader:
            self.fail()