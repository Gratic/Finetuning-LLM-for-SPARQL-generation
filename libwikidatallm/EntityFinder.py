from abc import ABC, abstractmethod
from typing import List
import requests

class EntityFinder(ABC):
    @abstractmethod
    def find_entities(self, name: str) -> List[tuple[str,str]]:
        pass

class PropertyFinder(ABC):
    @abstractmethod
    def find_properties(self, name: str) -> List[tuple[str,str]]:
        pass
    
class WikidataAPI(EntityFinder, PropertyFinder):
    def __init__(self, base_url: str = "https://www.wikidata.org/w/api.php") -> None:
        self.base_url = base_url
        
    def find_entities(self, name: str) -> List[tuple[str,str]]:
        payload = {
            "action": "wbsearchentities",
            "search": name,
            "language": "en",
            "format": "json"
        }
        data = requests.get(self.base_url, params=payload).json()
        
        items = data['search']
        results = [(item['id'], item['display']['label']['value']) for item in items]
        
        return results
    
    def find_properties(self, name: str) -> List[tuple[str,str]]:
        payload = {
            "action": "wbsearchentities",
            "search": name,
            "type": "property",
            "language": "en",
            "format": "json"
        }
        data = requests.get(self.base_url, params=payload).json()
        
        items = data['search']
        results = [(item['id'], item['display']['label']['value']) for item in items]
        
        return results