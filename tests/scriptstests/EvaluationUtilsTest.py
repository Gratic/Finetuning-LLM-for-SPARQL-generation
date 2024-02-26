import unittest
from modules.evaluation_utils import compute_recall, compute_precision, get_nested_values, average_precision, average_precision_slow
 
class EvaluationUtilsTest(unittest.TestCase):
    
    def test_compute_recall_exact_match(self):
        hyp = ["a", "b", "c"]
        labels = ["a", "b", "c"]
        
        self.assertEqual(1, compute_recall(hyp, labels))
    
    def test_compute_recall_no_match(self):
        hyp = []
        labels = ["a", "b", "c"]
        
        self.assertEqual(0, compute_recall(hyp, labels))
        
    def test_compute_recall_no_relevant(self):
        hyp = ["d", "e", "f"]
        labels = ["a", "b", "c"]
        
        self.assertEqual(0, compute_recall(hyp, labels))
    
    def test_compute_recall_one_relevant(self):
        hyp = ["a", "e", "f"]
        labels = ["a", "b", "c"]
        
        self.assertAlmostEqual(1./3., compute_recall(hyp, labels))
    
    def test_compute_precision_exact_match(self):
        hyp = ["a", "b", "c"]
        labels = ["a", "b", "c"]
        
        self.assertEqual(1, compute_precision(hyp, labels))
    
    def test_compute_precision_no_match(self):
        hyp = []
        labels = ["a", "b", "c"]
        
        self.assertEqual(0, compute_precision(hyp, labels))
    
    def test_compute_precision_one_relevant(self):
        hyp = ["a", "d"]
        labels = ["a", "b", "c"]
        
        self.assertEqual(0.5, compute_precision(hyp, labels))
    
    def test_compute_precision_no_relevant(self):
        hyp = ["d", "f", "e"]
        labels = ["a", "b", "c"]
        
        self.assertEqual(0, compute_precision(hyp, labels))
    
    def test_compute_precision_not_same_type(self):
        hyp = ["a", "b", "c"]
        label = [1, 2, 3]
        
        self.assertEqual(0, compute_precision(hyp, label))
    
    def test_compute_recall_not_same_type(self):
        hyp = ["a", "b", "c"]
        label = [1, 2, 3]
        
        self.assertEqual(0, compute_recall(hyp, label))
    
    def test_compute_precision_empty(self):
        hyp = []
        label = []
        
        self.assertEqual(1, compute_precision(hyp, label))
    
    def test_compute_recall_empty(self):
        hyp = []
        label = []
        
        self.assertEqual(1, compute_recall(hyp, label))
    
    def test_compute_precision_not_list(self):
        hyp = "a"
        label = []
        
        self.assertEqual(0, compute_precision(hyp, label))
    
    def test_compute_recall_not_list(self):
        hyp = "a"
        label = []
        
        self.assertEqual(0, compute_recall(hyp, label))
    
    def test_compute_precision_none(self):
        hyp = None
        label = None
        
        self.assertEqual(1., compute_precision(hyp, label))
    
    def test_compute_recall_none(self):
        hyp = None
        label = None
        
        self.assertEqual(1., compute_recall(hyp, label))
    
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

    def test_average_precision_1(self):
        hyp = [1, 2, 3]
        gold = [1, 2, 3]
        
        self.assertEqual(average_precision(hyp, gold), average_precision_slow(hyp, gold))
    
    def test_average_precision_2(self):
        hyp = [1]
        gold = [1, 2, 3]
        
        self.assertEqual(average_precision(hyp, gold), average_precision_slow(hyp, gold))
    
    def test_average_precision_3(self):
        hyp = []
        gold = [1, 2, 3]
        
        self.assertEqual(average_precision(hyp, gold), average_precision_slow(hyp, gold))
    
    def test_average_precision_4(self):
        hyp = [3, 2, 1]
        gold = [1, 2, 3]
        
        self.assertEqual(average_precision(hyp, gold), average_precision_slow(hyp, gold))
        
    def test_average_precision_5(self):
        hyp = [3, 2, 1]
        gold = []
        
        self.assertEqual(average_precision(hyp, gold), average_precision_slow(hyp, gold))
        
    def test_average_precision_6(self):
        hyp = []
        gold = []
        
        self.assertEqual(average_precision(hyp, gold), average_precision_slow(hyp, gold))
        
    def test_average_precision_none(self):
        hyp = None
        gold = None
        
        self.assertEqual(average_precision(hyp, gold), average_precision_slow(hyp, gold))