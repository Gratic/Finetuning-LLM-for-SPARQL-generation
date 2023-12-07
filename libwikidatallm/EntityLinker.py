from abc import ABC, abstractmethod
from typing import List
from .Pipeline import PipelineStep
from .EntityFinder import WikidataAPI

class EntityLinker(ABC):
    @abstractmethod
    def link_entities(self, extracted: List[str]) -> List[tuple[str, tuple[str,str]]]:
        pass
    
class PropertyLinker(ABC):
    @abstractmethod
    def link_properties(self, extracted: List[str]) -> List[tuple[str, tuple[str,str]]]:
        pass
    
class FirstWikidataEntityLinker(EntityLinker, PropertyLinker, PipelineStep):
    def __init__(self) -> None:
        self.wikidataAPI = WikidataAPI()
    
    def link_entities(self, extracted: List[str]) -> List[tuple[str, tuple[str,str]]]:
        entities_linked = []
        for entity in extracted:
            result = self.wikidataAPI.find_entities(entity)
            entities_linked.append((entity, result[0]))
        return entities_linked  

    def link_properties(self, extracted: List[str]) -> List[tuple[str, tuple[str,str]]]:
        properties_linked = []
        for entity in extracted:
            result = self.wikidataAPI.find_properties(entity)
            properties_linked.append((entity, result[0]))
        return properties_linked 
    
    def execute(self, context: dict):
        context["linked_entities"] = self.link_entities(context["extracted_entities"])
        context["linked_properties"] = self.link_entities(context["extracted_properties"])