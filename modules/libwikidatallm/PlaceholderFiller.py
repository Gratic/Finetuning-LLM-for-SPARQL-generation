# TODO: maybe delete

from abc import ABC, abstractmethod
from typing import List, Tuple
from .Pipeline import PipelineStep
import re

class PlaceholderFiller(ABC):
    @abstractmethod
    def deannotate(self, sparql: str, linked_entities: List[Tuple[str, str]], linked_properties: List[Tuple[str, str]]):
        pass
    
class SimplePlaceholderFiller(PlaceholderFiller, PipelineStep):
    def __init__(self, input_column_query: str = "translated_prompt", input_column_entities:str = "linked_entities", input_column_properties:str = "linked_properties", output_column:str = "linked_query") -> None:
        super().__init__()
        self.input_column_query = input_column_query
        self.input_column_entities = input_column_entities
        self.input_column_properties = input_column_properties
        self.output_column = output_column
    
    def deannotate(self, sparql: str, linked_entities: List[Tuple[str, Tuple[str,str]]], linked_properties: List[Tuple[str, Tuple[str,str]]]):
        for label, (entity_id, _) in linked_entities:
            sparql = re.sub(rf"\[entity:{label}\]", entity_id, sparql)
        
        for label, (entity_id, _) in linked_properties:
            sparql = re.sub(rf"\[property:{label}\]", entity_id, sparql)
        
        return sparql
    
    def execute(self, context: dict):
        context[self.output_column] = self.deannotate(context[self.input_column_query], context[self.input_column_entities], context[self.input_column_properties])    