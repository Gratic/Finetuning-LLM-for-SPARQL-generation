from abc import ABC, abstractmethod
from requests.exceptions import HTTPError, ConnectionError
from typing import List, Tuple
import requests
import time
from functools import lru_cache

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
    
    def _get_label_from_wbsearchentities(self, item):
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
    
    def _check_and_get_labels_from_wbsearchentities_response(self, response: requests.Response):
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

            results.append(self._get_label_from_wbsearchentities(item))
            
        return results
    
    def _retry_after_middle_man(self, func, num_retries:int = 3, **func_kwargs):
        is_error = True
        num_connection_retries = 3
        while num_retries > 0 and num_connection_retries > 0 and is_error:
            try:
                response = func(**func_kwargs)
                is_error = False
            except HTTPError as inst:
                if inst.response.status_code == 429:
                    retry_after = int(inst.response.headers['retry-after'])
                    time.sleep(retry_after + 1)
                    num_retries -= 1
                else:
                    raise inst
            except ConnectionError:
                num_connection_retries -= 1
                time.sleep((3-num_connection_retries)*10)
        return response
    
    def _smart_get_label_from_wbsearchentities(self, name:str, is_property:bool = False, num_recurrence = 1):
        if num_recurrence < 0:
            raise RecursionError("The recursion limit set has been exceeded. No name has been found.")
        
        response = self._retry_after_middle_man(self._get_response_from_wbsearchentities, num_retries=3, name=name, search_property=is_property)

        try:
            return self._check_and_get_labels_from_wbsearchentities_response(response=response)
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
    
    # This function get way more data than _get_response_from_wbsearchentities
    # but has the benefit of not caring if the id is entity or property.
    # Because Ids should be unique, it should works (obviously): https://www.wikidata.org/wiki/Wikidata:Identifiers
    def _get_response_from_entity_id(self, id:str):
        endpoint = "https://www.wikidata.org/entity/"
        response = requests.get(f"{endpoint}{id}", headers={'User-agent': 'WikidataLLM bot v0'}, allow_redirects=True)
        response.raise_for_status()
        
        return response
    
    def _check_and_get_labels_from_entity_response(self, response: requests.Response):
        try:
            data = response.json()
        except:
            raise NameError("The id doesn't exist.")
        
        results = []
        
        if 'entities' in data.keys():
            entities = data['entities']
        else:
            raise KeyError("No key entities in data.")
        
        for entity_id in entities.keys():
            if len(entities[entity_id]['labels']) == 0:
                raise NotImplementedError("The entity doesn't have any label.")
            
            labels = entities[entity_id]['labels']
            
            if 'en' in labels.keys():
                label = labels['en']['value']
            else:
                firstLanguage = list(labels.keys())[0]
                label = labels[firstLanguage]['value']
                print(f"The id={entity_id} doesn't have english labels. Taking the first in the list of labels ({firstLanguage} => {label}).")
            
            results.append((entity_id, label.strip()))
            
        return results
    
    @lru_cache(maxsize=32768)
    def _smart_get_labels_from_entity_id(self, name:str):
        try:
            response = self._retry_after_middle_man(self._get_response_from_entity_id, num_retries=3, id=name)
        except HTTPError as inst:
            # Id is incorrect
            return [(name, name)]
        
        return self._check_and_get_labels_from_entity_response(response)
            
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
        