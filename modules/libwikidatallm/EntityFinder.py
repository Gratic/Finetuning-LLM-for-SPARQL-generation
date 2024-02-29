from abc import ABC, abstractmethod
from requests.exceptions import HTTPError
from typing import List, Tuple
import requests
import time

class EntityFinder(ABC):
    @abstractmethod
    def find_entities(self, name: str) -> List[Tuple[str,str]]:
        pass

class PropertyFinder(ABC):
    @abstractmethod
    def find_properties(self, name: str) -> List[Tuple[str,str]]:
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
        
    def find_entities(self, name: str) -> List[Tuple[str,str]]:
        payload = {
            "action": "wbsearchentities",
            "search": name,
            "language": "en",
            "format": "json"
        }
        num_try = 3
        while num_try > 0:
            response = requests.get(self.base_url, params=payload, headers={'User-agent': 'WikidataLLM bot v0'})
            
            try:
                response.raise_for_status()
            except HTTPError as inst:
                if inst.response.status_code == 429:
                    retry_after = int(inst.response.headers['retry-after'])
                    time.sleep(retry_after + 1)
                    num_try -= 1
                    continue
                else:
                    raise inst
            
            break            
        data = response.json()
        
        items = data['search']
        
        if len(items) == 0:
            raise ValueError(f"The Wikidata API entity result returned empty with search={name}.")
        
        results = [(item['id'], item['display']['label']['value']) for item in items]
        
        return results
    
    def find_properties(self, name: str) -> List[Tuple[str,str]]:
        payload = {
            "action": "wbsearchentities",
            "search": name,
            "type": "property",
            "language": "en",
            "format": "json"
        }
        num_try = 3
        while num_try > 0:
            response = requests.get(self.base_url, params=payload, headers={'User-agent': 'WikidataLLM bot v0'})
            try:
                response.raise_for_status()
            except HTTPError as inst:
                if inst.response.status_code == 429:
                    retry_after = int(inst.response.headers['retry-after'])
                    time.sleep(retry_after + 1)
                    num_try -= 1
                    continue
                else:
                    raise inst
        
            break
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
        except requests.exceptions.JSONDecodeError:
            data = SPARQLResponse(response.text)
        
        return data
        