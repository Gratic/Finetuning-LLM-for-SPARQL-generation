from abc import ABC, abstractmethod
from requests.exceptions import HTTPError
from typing import List, Tuple
import requests
import time
import re

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
    
    def _recover_redirected_entity_id(self, name: str):
        if re.search(r"Q\d+", name):
            response = requests.get(f"http://www.wikidata.org/entity/{name}", allow_redirects=True)
            data = response.json()
            return list(data['entities'].keys())[0]
    
    def _get_labels(self, items):
        results = []
        for item in items:
            if 'label' in item['display'].keys():
                results.append((item['id'], item['display']['label']['value']))
            elif 'description' in item['display'].keys():
                results.append((item['id'], item['display']['description']['value']))
            else:
                raise NotImplementedError("Not implemented for case where there is no label or description.")
        
        return results
        
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
        
        if len(items[0]['display']) == 0:
            new_name = self._recover_redirected_entity_id(name)
            results = self.find_entities(new_name)
        else:
            try:
                results = self._get_labels(items)
            except Exception as inst:
                print(name)
                print(response.json())
                print(items)
                raise inst
        
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
        
        results = self._get_labels(items)
        
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
        