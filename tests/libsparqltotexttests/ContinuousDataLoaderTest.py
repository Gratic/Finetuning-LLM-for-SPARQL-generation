import unittest
from modules.libsparqltotext import ContinuousDataLoader
import pandas as pd

class ContinuousDataLoaderTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        
    def test_normal_case(self):
        value = [10, 20, 30, 40, 50]
        valueB = [100, 90, 80, 70, 60]
        dataset = pd.DataFrame()
        dataset["value"] = value
        dataset["valueB"] = valueB
        
        dataloader = ContinuousDataLoader(dataset, starting_row=0, last_row_index=len(value))
        
        count = 0
        for x in dataloader:
            self.assertEqual(x["value"], value[count], "Row order is wrong.")
            self.assertEqual(x["valueB"], valueB[count], "Row order is wrong.")
            count += 1
        
        self.assertEqual(len(dataset), count)
    
    def test_from_0_to_2(self):
        value = [10, 20, 30, 40, 50]
        valueB = [100, 90, 80, 70, 60]
        dataset = pd.DataFrame()
        dataset["value"] = value
        dataset["valueB"] = valueB
        
        test_value = [10, 20]
        test_valueB = [100, 90]
        
        dataloader = ContinuousDataLoader(dataset, starting_row=0, last_row_index=2)
        
        count = 0
        for x in dataloader:
            self.assertEqual(x["value"], test_value[count], "Row order is wrong.")
            self.assertEqual(x["valueB"], test_valueB[count], "Row order is wrong.")
            count += 1
        
        self.assertEqual(2, count)
        
    def test_from_1_to_2(self):
        value = [10, 20, 30, 40, 50]
        valueB = [100, 90, 80, 70, 60]
        dataset = pd.DataFrame()
        dataset["value"] = value
        dataset["valueB"] = valueB
        
        test_value = [20]
        test_valueB = [90]
        
        dataloader = ContinuousDataLoader(dataset, starting_row=1, last_row_index=2)
        
        count = 0
        for x in dataloader:
            self.assertEqual(x["value"], test_value[count], "Row order is wrong.")
            self.assertEqual(x["valueB"], test_valueB[count], "Row order is wrong.")
            count += 1
        
        self.assertEqual(1, count)
        
    def test_from_1_to_5(self):
        value = [10, 20, 30, 40, 50]
        valueB = [100, 90, 80, 70, 60]
        dataset = pd.DataFrame()
        dataset["value"] = value
        dataset["valueB"] = valueB
        
        test_value = [20, 30, 40, 50]
        test_valueB = [90, 80, 70, 60]
        
        dataloader = ContinuousDataLoader(dataset, starting_row=1, last_row_index=5)
        
        count = 0
        for x in dataloader:
            self.assertEqual(x["value"], test_value[count], "Row order is wrong.")
            self.assertEqual(x["valueB"], test_valueB[count], "Row order is wrong.")
            count += 1
        
        self.assertEqual(4, count)
    
    def test_starting_row_is_above_last_row_index(self):
        value = [10, 20, 30, 40, 50]
        valueB = [100, 90, 80, 70, 60]
        dataset = pd.DataFrame()
        dataset["value"] = value
        dataset["valueB"] = valueB
        
        dataloader = ContinuousDataLoader(dataset, starting_row=2, last_row_index=1)
        
        for x in dataloader:
            self.fail()
    
    def test_starting_row_is_negative(self):
        with self.assertRaises(ValueError):
            ContinuousDataLoader(pd.DataFrame(), -5, 5)
            
    def test_last_row_index_is_negative(self):
        value = [10, 20, 30, 40, 50]
        valueB = [100, 90, 80, 70, 60]
        dataset = pd.DataFrame()
        dataset["value"] = value
        dataset["valueB"] = valueB
        
        dataloader = ContinuousDataLoader(dataset, starting_row=2, last_row_index=-5)
        
        for x in dataloader:
            self.fail()
    
    def test_dataset_is_not_a_pandas_dataframe(self):
        dataset = []
        
        with self.assertRaises(TypeError):
            ContinuousDataLoader(dataset, starting_row=0, last_row_index=1)