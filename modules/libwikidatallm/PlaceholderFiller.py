# TODO: maybe delete

from abc import ABC, abstractmethod
from typing import List, Tuple
from .Pipeline import PipelineStep

class PlaceholderFiller(ABC):
    @abstractmethod
    def deannotate(self, sparql: str, linked_entities: List[Tuple[str, str]], linked_properties: List[Tuple[str, str]]):
        pass
    
class SimplePlaceholderFiller(PlaceholderFiller, PipelineStep):
    def deannotate(self, sparql: str, linked_entities: List[Tuple[str, Tuple[str,str]]], linked_properties: List[Tuple[str, Tuple[str,str]]]):
        for i, (_, (linked_label, _)) in enumerate(linked_entities):
            sparql = sparql.replace(f"[entity {i}]", linked_label)
        
        for i, (_, (linked_label, _)) in enumerate(linked_properties):
            sparql = sparql.replace(f"[property {i}]", linked_label)
        
        return sparql
    
    def execute(self, context: dict):
        context["deannotated_sentence"] = self.deannotate(context["translated_prompt"], context["linked_entities"], context["linked_properties"])    