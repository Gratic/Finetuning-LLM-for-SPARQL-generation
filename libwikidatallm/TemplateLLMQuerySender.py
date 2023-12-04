from .LLMConnector import LLMConnector, LLMResponse
from typing import List

class TemplateLLMQuerySender():
    def __init__(self, llm: LLMConnector, template_text: str = "") -> None:
        self.llm = llm
        self.template_text = template_text
    
    def completion(self, data: dict[str, str]) -> LLMResponse:
        prompt = self.apply_template(data)
        
        return self.llm.completion(prompt)

    def tokenize(self, data: dict[str, str]) -> List[int]:
        prompt = self.apply_template(data)
        
        return self.llm.tokenize(prompt)
    
    def apply_template(self, data: dict[str, str]):
        prompt = self.template_text
        
        for k, v in data.items():
            prompt.replace(k, v)
        return prompt