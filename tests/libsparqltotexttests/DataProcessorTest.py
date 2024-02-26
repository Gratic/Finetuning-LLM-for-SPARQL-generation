from typing import List
import unittest
from modules.libsparqltotext import DataProcessor, BaseProvider, BaseAnswerProcessor
import pandas as pd

class MockupProvider(BaseProvider):
    def __init__(self) -> None:
        super().__init__()
        self.test_case = 0
    
    def query(self, parameters: dict[str, str | int | float]) -> bool:
        if self.test_case == 0:
            answer ="This is my answer: a very long and automatically generated text. I need further processing though."
            
            self.last_answer = answer
            self.last_full_answer = answer
            return answer
            
    def get_tokens(self, parameters: dict[str, str | int | float]) -> List[int]:
        pass

class MockupAnswerProcessor(BaseAnswerProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.test_case = 0
        self.number_of_retry = 0
    
    def get_prompts(self, generated_text: str) -> List[str]:
        if self.test_case == 0:
            return ["Prompt 1", "Prompt 2", "Prompt 3"]
        if self.test_case == 1:
            self.number_of_retry += 1
            return []

class DataProcessorTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
    
    def setUp(self) -> None:
        dataset = pd.DataFrame()
        dataset["query"] = pd.Series(data=["my query", "this query is very long", "", "", ""])
        dataset["description"] = pd.Series(data=["my description", "this description is very long", "", "", ""])
        dataset["context"] = pd.Series(data=["my context", "this context is very long", "", "", ""])
        dataset["prompt"] = pd.Series(data=["my prompt", "this prompt is very long", "", "", ""])
        dataset["num_tokens"] = pd.Series(data=[150, 99999999, 4096, 4095, 4097])
        dataset["result"] = pd.Series(data=[None, None, None, None, None])
        dataset["full_answer"] = pd.Series(data=[None, None, None, None, None])
        dataset["is_skipped"] = pd.Series(data=[None, None, None, None, None])
        dataset["is_prompt_too_long"] = pd.Series(data=[None, None, None, None, None])
        
        self.mockupProvider = MockupProvider()
        self.mockupAnswerProcessor = MockupAnswerProcessor()
        self.dataProcessor = DataProcessor(self.mockupProvider, self.mockupAnswerProcessor, dataset, 10, 4096, False, False)
        
    def test_process_row_valid(self):
        self.mockupProvider.test_case = 0
        self.mockupAnswerProcessor.test_case = 0
        self.assertEqual(self.dataProcessor.process_row_number(0), (["Prompt 1", "Prompt 2", "Prompt 3"], "This is my answer: a very long and automatically generated text. I need further processing though.", False, False))
        
    def test_process_row_prompt_far_too_long(self):
        self.assertEqual(self.dataProcessor.process_row_number(1), (None, None, True, True))
        
    def test_process_row_prompt_exact_context_length(self):
        self.mockupProvider.test_case = 0
        self.mockupAnswerProcessor.test_case = 0
        self.assertEqual(self.dataProcessor.process_row_number(2), (["Prompt 1", "Prompt 2", "Prompt 3"], "This is my answer: a very long and automatically generated text. I need further processing though.", False, False))
        
    def test_process_row_prompt_a_bit_shorter(self):
        self.mockupProvider.test_case = 0
        self.mockupAnswerProcessor.test_case = 0
        self.assertEqual(self.dataProcessor.process_row_number(3), (["Prompt 1", "Prompt 2", "Prompt 3"], "This is my answer: a very long and automatically generated text. I need further processing though.", False, False))
        
    def test_process_row_prompt_a_bit_too_long(self):
        self.assertEqual(self.dataProcessor.process_row_number(4), (None, None, True, True))
        
    def test_process_row_too_many_tries(self):
        self.mockupProvider.test_case = 0
        self.mockupAnswerProcessor.test_case = 1
        number_of_retry = 10
        self.dataProcessor.retry_attempts = number_of_retry
        
        self.assertEqual(self.dataProcessor.process_row_number(0), (None, None, True, False))
        self.assertEqual(self.mockupAnswerProcessor.number_of_retry, number_of_retry)
            
    def test_are_results_acceptable_valid(self):
        results = ["answer 1", "answer 2", "answer 3"]
        banned_words = ["ban"]
        self.assertTrue(DataProcessor.are_results_acceptable(results, banned_words))
    
    def test_are_results_acceptable_empty_not_valid(self):
        results = []
        banned_words = ["ban"]
        self.assertFalse(DataProcessor.are_results_acceptable(results, banned_words))
        
    def test_are_results_acceptable_banned_word_not_valid(self):
        results = ["ban"]
        banned_words = ["ban"]
        self.assertFalse(DataProcessor.are_results_acceptable(results, banned_words))
        
    def test_are_results_acceptable_banned_word_among_many_answer_not_valid(self):
        results = ["answer 1", "this sentence uses the ban word !!!"]
        banned_words = ["ban"]
        self.assertFalse(DataProcessor.are_results_acceptable(results, banned_words))