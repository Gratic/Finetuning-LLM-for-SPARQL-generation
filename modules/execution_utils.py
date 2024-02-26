from constants import PREFIX_TO_URL
import re

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