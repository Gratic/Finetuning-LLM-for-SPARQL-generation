import re
from abc import ABC, abstractmethod
from typing import List

class BaseAnswerProcessor(ABC):
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def get_prompts(self, generated_text: str) -> List[str]:
        pass
class RegexAnswerProcessor(BaseAnswerProcessor):
    def __init__(self, args) -> None:
        if args.verbose:
            print("Starting execution.")
            print("Compiling regex... ", end="")
        
        self.pattern = re.compile(r'\"[A-Z].*\"', flags=0)
        
        if args.verbose:
            print("Done.")
    
    def get_prompts(self, generated_text):
        return self.pattern.findall(generated_text)