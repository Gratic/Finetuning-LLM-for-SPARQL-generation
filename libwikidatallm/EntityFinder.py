from abc import ABC, abstractmethod
from typing import List
import requests

class EntityFinder(ABC):
    @abstractmethod
    def find_entities(self, name: str) -> List[str]:
        pass

class PropertyFinder(ABC):
    @abstractmethod
    def find_properties(self, name: str) -> List[str]:
        pass
    
class WikidataAPI(EntityFinder, PropertyFinder):
    def __init__(self) -> None:
        super().__init__()
        
    def find_entities(self, name: str) -> List[tuple[str,str]]:
        data = requests.get(f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={name}&language=en&format=json").json()
        
        items = data['search']
        results = [(item['id'], item['display']['label']['value']) for item in items]
        
        return results
    
    def find_properties(self, name: str) -> List[tuple[str,str]]:
        data = requests.get(f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={name}&language=en&format=json&type=property").json()
        items = data['search']
        results = [(item['id'], item['display']['label']['value']) for item in items]
        
        return results