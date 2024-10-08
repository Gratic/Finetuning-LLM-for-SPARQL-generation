import sys
from pathlib import Path
sys.path.append(Path("modules").absolute().__str__())

from .LLMConnector import LLMConnector, LLMResponse
from prompts_template import LLAMA2_TEMPLATE
from typing import List, Dict

class TemplateLLMQuerySender():
    def __init__(self, llm: LLMConnector, template_text: str = LLAMA2_TEMPLATE, start_seq="", end_seq="") -> None:
        self.llm = llm
        self.template_text = template_text
        self.start_seq = start_seq
        self.end_seq = end_seq
    
    def completion(self, data: Dict[str, str]) -> LLMResponse:
        prompt = self.apply_template(data)
        
        return self.llm.completion(prompt)

    def tokenize(self, data: Dict[str, str]) -> List[int]:
        prompt = self.apply_template(data)
        
        return self.llm.tokenize(prompt)
    
    def apply_template(self, data: Dict[str, str]):
        if not isinstance(data, dict):
            raise TypeError("data must be a dictionnary.")
        
        prompt = self.template_text
        
        for k, v in data.items():
            if not isinstance(k, str):
                raise TypeError("All keys must be strings.")
            
            if not isinstance(v, str):
                v = str(v)
            
            prompt = prompt.replace(f"{self.start_seq}{k}{self.end_seq}", v)
        return prompt