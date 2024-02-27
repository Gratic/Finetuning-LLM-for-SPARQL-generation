import sys
from pathlib import Path
sys.path.append(Path("modules").absolute().__str__())

from constants import PREFIX_TO_URL
import re
from requests.exceptions import HTTPError, Timeout
import time
from libwikidatallm.EntityFinder import WikidataAPI

def is_query_empty(query :str) -> bool:
    return query is None or query.strip() == "" or len(query.strip()) == 0

def can_add_limit_clause(query :str) -> bool:
    upper_query = query.upper()
    return (not is_query_empty(query) and not re.search(r"\WCOUNT\W", upper_query) and not re.search(r"\WLIMIT\W", upper_query))

def add_relevant_prefixes_to_query(query: str):
    prefixes = ""
    copy_query = query
    for k in PREFIX_TO_URL.keys():
        current_prefix = f"PREFIX {k}: <{PREFIX_TO_URL[k]}>"
        
        # Some queries already have some prefixes, duplicating them will cause an error
        # So first we check that the prefix we want to add is not already included.
        if not re.search(current_prefix, copy_query): 
            
            # Then we look for the prefix in the query
            if re.search(rf"\W({k}):", copy_query):
                prefixes += current_prefix + "\n"
        
        # For safety, we remove all the constants that starts with the prefix
        while re.search(rf"\W({k}):", copy_query):
            copy_query = re.sub(rf"\W({k}):", " ", copy_query)
    
    if prefixes != "":
        prefixes += "\n"
    
    return prefixes + query

def send_query_to_api(query, api, timeout_limit, num_try, do_print=True):
    response = None
    while num_try > 0 and response == None and not is_query_empty(query):
        try:
            if do_print:
                print(f"| Calling API... ", end="", flush=True)
            sparql_response = api.execute_sparql(query, timeout=timeout_limit)
            response = sparql_response.bindings if sparql_response.success else sparql_response.data
                
        except HTTPError as inst:
            if inst.response.status_code == 429:
                retry_after = int(inst.response.headers['retry-after'])
                if do_print:
                    print(f"| Retry-after: {retry_after} ", end="", flush=True)
                time.sleep(retry_after + 1)
                num_try -= 1
            else:
                if do_print:
                    print(f"| Exception occured ", end="", flush=True)
                response = "exception: " + str(inst) + "\n" + inst.response.text
        except Timeout:
            response = "timeout"
            if do_print:
                print(f"| Response Timeout ", end="", flush=True)
        except Exception as inst:
            if do_print:
                print(f"| Exception occured ", end="", flush=True)
            response = "exception: " + str(inst)
    return response if response != None else "exception: too many retry-after"

def prepare_and_send_query_to_api(query, index, num_of_rows, api=WikidataAPI(), answer_limit=10, timeout_limit=60, do_add_limit=True, do_print=True):
    if do_print:
        print(f"row {str(index)}/{num_of_rows} ".ljust(15), end="", flush=True)
    response = None
        
    if is_query_empty(query):
        response = "exception: query is empty"
        if do_print:
            print(f"| Query is empty ", end="", flush=True)
    else:
        query = add_relevant_prefixes_to_query(query)
            
        if do_add_limit and can_add_limit_clause(query):
            query += f"\nLIMIT {answer_limit}"
            
            
        response = send_query_to_api(query=query,
                                    api=api,
                                    timeout_limit=timeout_limit,
                                    num_try=3,
                                    do_print=do_print)
    if do_print:
        print(f"| done.", flush=True)
    return query, response