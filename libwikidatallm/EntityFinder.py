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

class SPARQLQueryEngine(ABC):
    @abstractmethod
    def execute_sparql(self, query: str):
        pass
    
class SPARQLResponse():
    def __init__(self, data) -> None:
        self.data = data
        if isinstance(data, dict):
            if "results" in data and "bindings" in data["results"]:
                self.bindings = data['results']['bindings']
                self.success = True
        else:
            self.bindings = False
            self.success = False
class WikidataAPI(EntityFinder, PropertyFinder, SPARQLQueryEngine):
    def __init__(self, base_url: str = "https://www.wikidata.org/w/api.php") -> None:
        self.base_url = base_url
        
    def find_entities(self, name: str) -> List[tuple[str,str]]:
        payload = {
            "action": "wbsearchentities",
            "search": name,
            "language": "en",
            "format": "json"
        }
        response = requests.get(self.base_url, params=payload, headers={'User-agent': 'WikidataLLM bot v0'})
        response.raise_for_status()
        
        data = response.json()
        
        items = data['search']
        
        if len(items) == 0:
            raise ValueError(f"The Wikidata API entity result returned empty with search={name}.")
        
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
        response = requests.get(self.base_url, params=payload, headers={'User-agent': 'WikidataLLM bot v0'})
        response.raise_for_status()
        
        data = response.json()
        
        items = data['search']
        
        if len(items) == 0:
            raise ValueError(f"The Wikidata API property result returned empty with search={name}.")
        
        results = [(item['id'], item['display']['label']['value']) for item in items]
        
        return results
    
    def execute_sparql(self, query: str, timeout: int = None):
        url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
        response = requests.get(url, params={'query': query, 'format': 'json'}, headers={'User-agent': 'WikidataLLM bot v0'}, timeout=timeout)
        response.raise_for_status()
        
        try:
            data = SPARQLResponse(response.json())
        except requests.exceptions.JSONDecodeError as inst:
            data = SPARQLResponse(response.text)
        
        return data
        