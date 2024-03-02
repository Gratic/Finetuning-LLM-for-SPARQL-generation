# TODO: maybe delete

from abc import ABC, abstractmethod
from typing import List
from .Pipeline import PipelineStep

class SentencePlaceholder(ABC):
    
    @abstractmethod
    def annotate(self, sentence: str, entities: List[str], properties: List[str]) -> str:
        pass
    
class SimpleSentencePlaceholder(SentencePlaceholder, PipelineStep):
    def annotate(self, sentence: str, entities: List[str], properties: List[str]) -> str:
        for i, entity in enumerate(entities):
            sentence = sentence.replace(entity, f"[entity {i}]")
        
        for i, property in enumerate(properties):
            sentence = sentence.replace(property, f"[property {i}]")
        
        return sentence
    
    def execute(self, context: dict):
        context["annotated_sentence"] = self.annotate(context["row"], context["extracted_entities"], context["extracted_properties"])    