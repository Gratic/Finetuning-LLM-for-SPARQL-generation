# TODO: maybe delete

from abc import ABC, abstractmethod
from typing import List, Tuple
from .Pipeline import PipelineStep
from .EntityFinder import WikidataAPI

class EntityLinker(ABC):
    @abstractmethod
    def link_entities(self, extracted: List[str]) -> List[Tuple[str, Tuple[str,str]]]:
        pass
    
class PropertyLinker(ABC):
    @abstractmethod
    def link_properties(self, extracted: List[str]) -> List[Tuple[str, Tuple[str,str]]]:
        pass
    
class TakeFirstWikidataEntityLinker(EntityLinker, PropertyLinker, PipelineStep):
    def __init__(self, input_column_entities:str = 'extracted_entities', input_column_properties:str = 'extracted_properties', output_column_entities:str = 'linked_entities', output_column_properties:str = 'linked_properties') -> None:
        self.wikidataAPI = WikidataAPI()
        self.input_column_entities = input_column_entities
        self.input_column_properties = input_column_properties
        self.output_column_entities = output_column_entities
        self.output_column_properties = output_column_properties
    
    def link_entities(self, extracted: List[str]) -> List[Tuple[str, Tuple[str,str]]]:
        entities_linked = []
        for entity in extracted:
            result = self.wikidataAPI.find_entities(entity)
            entities_linked.append((entity, result[0]))
        return entities_linked  

    def link_properties(self, extracted: List[str]) -> List[Tuple[str, Tuple[str,str]]]:
        properties_linked = []
        for entity in extracted:
            result = self.wikidataAPI.find_properties(entity)
            properties_linked.append((entity, result[0]))
        return properties_linked 
    
    def execute(self, context: dict):
        context[self.output_column_entities] = self.link_entities(context[self.input_column_entities])
        context[self.output_column_properties] = self.link_properties(context[self.input_column_properties])