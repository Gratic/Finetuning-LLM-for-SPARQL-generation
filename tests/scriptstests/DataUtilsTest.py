import sys
from pathlib import Path
sys.path.append(Path("modules").absolute().__str__())

import unittest
from modules.data_utils import get_nested_values
 
class DataUtilsTest(unittest.TestCase): 
    def test_get_nested_values_nested_tree(self):
        tree = {
            "char1": "non-important",
            "value": "very important",
            "child": {
                "cahr2": "forgettable",
                "value": "non forgettable",
                "child": {
                    "value": "interesting"
                }
            }
        }
        result = ["very important", "non forgettable", "interesting"]
        
        self.assertListEqual(result, get_nested_values(tree))
    
    def test_get_nested_values_empty_tree(self):
        tree = {}
        result = []
        
        self.assertListEqual(result, get_nested_values(tree))
    
    def test_get_nested_values_single_element(self):
        tree = {"value": "oui"}
        result = ["oui"]
        
        self.assertListEqual(result, get_nested_values(tree))
        
    def test_get_nested_values_no_values(self):
        tree = {
            "element1": "oui",
            "element2": "non"
            }
        result = []
        
        self.assertListEqual(result, get_nested_values(tree))
    
    def test_get_nested_values_list_of_str(self):
        tree = ["a", "b", "c"]
        
        with self.assertRaises(TypeError):
            get_nested_values(tree)
            
    def test_get_nested_values_empty_list(self):
        tree = []
        result = []
        
        self.assertListEqual(result, get_nested_values(tree))
            
    def test_get_nested_values_list_of_trees(self):
        tree = [
            {"value": "one"},
            {"value": "two"},
            {"value": "three"}
            ]
        result = ["one", "two", "three"]
        
        self.assertListEqual(result, get_nested_values(tree))
            
    def test_get_nested_values_list_of_nested_trees(self):
        tree = [
            {
                "char1": 
                {
                    "char2": 
                        {
                            "value": "one"
                        }
                }
            },
            {
                "value": "two",
                "children": 
                    {
                    "value": "two-one"
                }
            },
            {
                "value": "three"
            }
            ]
        result = ["one", "two", "two-one", "three"]
        
        self.assertListEqual(result, get_nested_values(tree))