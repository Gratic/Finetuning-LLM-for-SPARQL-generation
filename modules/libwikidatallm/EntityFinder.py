from abc import ABC, abstractmethod
from requests.exceptions import HTTPError
from typing import List, Tuple
import requests
import time

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
    def execute_sparql(self, query: str) -> SPARQLResponse:
        pass
            
class WikidataAPI(EntityFinder, PropertyFinder, SPARQLQueryEngine):
    def __init__(self, base_url: str = "https://www.wikidata.org/w/api.php") -> None:
        self.base_url = base_url
    
    def _recover_redirected_id(self, name: str, is_property:bool = False):
        if is_property:
            print(f"{name=}")
            raise NotImplementedError("Not implemented for property yet.")
        
        endpoint = "http://www.wikidata.org/"
        endpoint += "property/" if is_property else "entity/"
        
        response = requests.get(f"{endpoint}{name}", allow_redirects=True)
        data = response.json()
        return list(data['entities'].keys())[0]
    
    def _get_label(self, item):
        if 'label' in item['display'].keys():
            return (item['id'], item['display']['label']['value'])
        elif 'description' in item['display'].keys():
            return (item['id'], item['display']['description']['value'])
        else:
            raise NotImplementedError("Not implemented for case where there is no label or description.")
    
    def _get_response_from_wbsearchentities(self, name: str, search_property: bool = False):
        payload = {
            "action": "wbsearchentities",
            "search": name,
            "language": "en",
            "format": "json"
        }
        
        if search_property:
            payload.update({"type": "property"})
        
        response = requests.get(self.base_url, params=payload, headers={'User-agent': 'WikidataLLM bot v0'})
        response.raise_for_status()
        return response
    
    def _check_and_get_labels_from_response(self, response):
        data = response.json()
        
        if not 'search' in data.keys():
            raise KeyError("The search key is not in the response data.")
        
        items = data["search"]
        
        if len(items) == 0:
            raise Exception("There is no results.")
        
        results = []
        for item in items:
            if not 'display' in item.keys():
                raise KeyError("There is no 'display' in item results.")
            
            if len(item['display']) == 0:
                raise NameError("It has been redirected.")

            results.append(self._get_label(item))
            
        return results
    
    def _retry_after_middle_man(self, name: str, search_property:bool = False, num_retries:int = 3):
        is_error = True
        while num_retries > 0 and is_error:
            try:
                response = self._get_response_from_wbsearchentities(name, search_property)
                is_error = False
            except HTTPError as inst:
                if inst.response.status_code == 429:
                    retry_after = int(inst.response.headers['retry-after'])
                    time.sleep(retry_after + 1)
                    num_retries -= 1
                else:
                    raise inst
        return response
    
    def _smart_get_label_from_wbsearchentities(self, name:str, is_property:bool = False, num_recurrence = 1):
        if num_recurrence < 0:
            raise RecursionError("The recursion limit set has been exceeded. No name has been found.")
        
        response = self._retry_after_middle_man(name, search_property=is_property, num_retries=3)

        try:
            return self._check_and_get_labels_from_response(response=response)
        except KeyError as inst:
            raise inst
        except NameError:
            name = self._recover_redirected_id(name, is_property=is_property)
            try:
                return self._smart_get_label_from_wbsearchentities(name, is_property=is_property, num_recurrence=num_recurrence-1)
            except RecursionError:
                return [(name, name)]
            except Exception as inst:
                raise inst
        except Exception:
            try:
                return self._smart_get_label_from_wbsearchentities(name, is_property=(not is_property), num_recurrence=num_recurrence-1)
            except RecursionError:
                return [(name, name)]
            except Exception as inst:
                raise inst
    
    def find_entities(self, name: str) -> List[Tuple[str,str]]:
        return self._smart_get_label_from_wbsearchentities(name, is_property=False)
    
    def find_properties(self, name: str) -> List[Tuple[str,str]]:
        return self._smart_get_label_from_wbsearchentities(name, is_property=True)
    
    def execute_sparql(self, query: str, timeout: int = None) -> SPARQLResponse:
        url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
        response = requests.get(url, params={'query': query, 'format': 'json'}, headers={'User-agent': 'WikidataLLM bot v0'}, timeout=timeout)
        response.raise_for_status()
        
        try:
            data = SPARQLResponse(response.json())
        except requests.exceptions.JSONDecodeError:
            data = SPARQLResponse(response.text)
        
        return data
        