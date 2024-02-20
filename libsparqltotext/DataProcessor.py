from typing import List
from .Provider import BaseProvider
import pandas as pd
from .AnswerProcessor import BaseAnswerProcessor
from typing import Union, Dict

RETRY_IF_ANSWER_CONTAINS = ["SELECT", "GROUP"]

class DataProcessor():
    def __init__(self, provider: BaseProvider, answerProcessor: BaseAnswerProcessor, dataset: pd.DataFrame, retry_attempts: int, context_length_limit: int, print_answers: bool, print_results: bool) -> None:
        self.provider: BaseProvider = provider
        self.answerProcessor: BaseAnswerProcessor = answerProcessor
        self.dataset: pd.DataFrame = dataset
        self.retry_attempts: int = retry_attempts
        self.context_length_limit: int = context_length_limit
        self.print_answers: bool = print_answers
        self.print_results: bool = print_results
        
        self.prompts: pd.Series = dataset['prompt']
        self.num_tokens: pd.Series = dataset['num_tokens']
    
    def process_row_number(self, row_index: int):
        '''Returns (results, full answer, skipped, context length too long)'''
        num_token = self.num_tokens.iat[row_index]
        
        if num_token > self.context_length_limit:
            return (None, None, True, True)
        
        number_of_try_left = self.retry_attempts
        while number_of_try_left != 0:    
            self.provider.query(self.prompts.iat[row_index])
            
            if self.print_answers:
                print(self.provider.get_answer())
            
            results = self.answerProcessor.get_prompts(self.provider.get_answer())
            
            if self.print_results:
                print(results)
            
            if not self.are_results_acceptable(results, RETRY_IF_ANSWER_CONTAINS):
                number_of_try_left -= 1
                continue
            
            return (results, self.provider.get_full_answer(), False, False)

        return (None, None, True, False)

    @staticmethod
    def are_results_acceptable(results: List[str], banned_words: List[str]) -> bool:
        is_good_quality = True
        
        if len(results) == 0:
            return False
        
        for result in results:
            for word in banned_words:
                if word in result:
                    is_good_quality = False
        return is_good_quality