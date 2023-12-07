from abc import ABC, abstractmethod
from typing import List
from .Pipeline import PipelineStep

class SentencePlaceholder(ABC):
    @abstractmethod
    def deannotate(self, sparql: str, linked_entities: List[tuple[str, str]], linked_properties: List[tuple[str, str]]):
        pass
    
class SimpleSentencePlaceholder(SentencePlaceholder):
    def deannotate(self, sparql: str, linked_entities: List[tuple[str, tuple[str,str]]], linked_properties: List[tuple[str, tuple[str,str]]]):
        for i, (_, (linked_label, _)) in enumerate(linked_entities):
            sparql = sparql.replace(f"[entity {i}]", linked_label)
        
        for i, (_, (linked_label, _)) in enumerate(linked_properties):
            sparql = sparql.replace(f"[property {i}]", linked_label)
        
        return sparql
    
    def execute(self, context: dict):
        context["deannotated_sentence"] = self.deannotate(context["translated_prompt"], context["linked_entities"], context["linked_properties"])    