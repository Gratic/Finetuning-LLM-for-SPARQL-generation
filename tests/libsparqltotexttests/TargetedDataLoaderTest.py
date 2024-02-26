import unittest
from modules.libsparqltotext import TargetedDataLoader
import pandas as pd

class TargetedDataLoaderTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        
    def test_normal(self):
        value = [10, 20, 30, 40, 50]
        valueB = [100, 90, 80, 70, 60]
        targets = [0, 2, 3]
        test_value = [10, 30, 40]
        test_valueB = [100, 80, 70]
        
        dataset = pd.DataFrame()
        dataset["value"] = value
        dataset["valueB"] = valueB
        
        dataloader = TargetedDataLoader(dataset, targets)
        
        count = 0
        for x in dataloader:
            self.assertEqual(x["value"], test_value[count], "Row order is not correct.")
            self.assertEqual(x["valueB"], test_valueB[count], "Row order is not correct.")
            count += 1
        
        self.assertEqual(count, len(targets))
    
    def test_targets_is_empty(self):
        value = [10, 20, 30, 40, 50]
        valueB = [100, 90, 80, 70, 60]
        targets = []
        test_value = [10, 30, 40]
        test_valueB = [100, 80, 70]
        
        dataset = pd.DataFrame()
        dataset["value"] = value
        dataset["valueB"] = valueB
        
        dataloader = TargetedDataLoader(dataset, targets)
        
        count = 0
        for x in dataloader:
            self.fail()
        
        self.assertEqual(count, len(targets))
    
    def test_targets_contains_index_out_of_dataset(self):
        value = [10, 20, 30, 40, 50]
        valueB = [100, 90, 80, 70, 60]
        targets = [5]
        
        dataset = pd.DataFrame()
        dataset["value"] = value
        dataset["valueB"] = valueB
        
        dataloader = TargetedDataLoader(dataset, targets)
        
        count = 0
        with self.assertRaises(IndexError):
            for x in dataloader:
                continue
    
    def test_targets_is_not_a_list(self):
        targets = "oops"
        
        with self.assertRaises(TypeError):
            TargetedDataLoader(pd.DataFrame, targets)
    
    def test_targets_is_a_list_but_not_of_int(self):
        targets = ["o", "o", "p", "s"]
        
        with self.assertRaises(TypeError):
            TargetedDataLoader(pd.DataFrame, targets)