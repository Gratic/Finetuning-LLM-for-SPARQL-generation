from libwikidatallm.EntityFinder import WikidataAPI
import pandas as pd
from requests.exceptions import HTTPError, Timeout
import time
import argparse
import os
import re

PREFIX_TO_URL = {
    # Prefixes from https://www.mediawiki.org/wiki/Special:MyLanguage/Wikibase/Indexing/RDF_Dump_Format#Full_list_of_prefixes
    "bd": "http://www.bigdata.com/rdf#",
    "cc": "http://creativecommons.org/ns#",
    "dct": "http://purl.org/dc/terms/",
    "geo": "http://www.opengis.net/ont/geosparql#",
    "hint": "http://www.bigdata.com/queryHints#" ,
    "ontolex": "http://www.w3.org/ns/lemon/ontolex#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "prov": "http://www.w3.org/ns/prov#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "schema": "http://schema.org/",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",

    "p": "http://www.wikidata.org/prop/",
    "pq": "http://www.wikidata.org/prop/qualifier/",
    "pqn": "http://www.wikidata.org/prop/qualifier/value-normalized/",
    "pqv": "http://www.wikidata.org/prop/qualifier/value/",
    "pr": "http://www.wikidata.org/prop/reference/",
    "prn": "http://www.wikidata.org/prop/reference/value-normalized/",
    "prv": "http://www.wikidata.org/prop/reference/value/",
    "psv": "http://www.wikidata.org/prop/statement/value/",
    "ps": "http://www.wikidata.org/prop/statement/",
    "psn": "http://www.wikidata.org/prop/statement/value-normalized/",
    "wd": "http://www.wikidata.org/entity/",
    "wdata": "http://www.wikidata.org/wiki/Special:EntityData/",
    "wdno": "http://www.wikidata.org/prop/novalue/",
    "wdref": "http://www.wikidata.org/reference/",
    "wds": "http://www.wikidata.org/entity/statement/",
    "wdt": "http://www.wikidata.org/prop/direct/",
    "wdtn": "http://www.wikidata.org/prop/direct-normalized/",
    "wdv": "http://www.wikidata.org/value/",
    "wikibase": "http://wikiba.se/ontology#",
    
    # Manually added prefixes
    "var_muntype": "http://www.wikidata.org/entity/Q15284",
    "var_area": "http://www.wikidata.org/entity/Q6308",
    "lgdo": "http://linkedgeodata.org/ontology/",
    "geom": "http://geovocab.org/geometry#",
    "bif": "bif:",
    "wp": "http://vocabularies.wikipathways.org/wp#",
    "dcterms": "http://purl.org/dc/terms/",
    "gas": "http://www.bigdata.com/rdf/gas#",
    "void": "http://rdfs.org/ns/void#",
    "pav": "http://purl.org/pav/",
    "freq": "http://purl.org/cld/freq/",
    "biopax": "http://www.biopax.org/release/biopax-level3.owl#",
    "gpml": "http://vocabularies.wikipathways.org/gpml#",
    "wprdf": "http://rdf.wikipathways.org/",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "vrank": "http://purl.org/voc/vrank#",
    "nobel": "http://data.nobelprize.org/terms/",
    "dbc": "http://dbpedia.org/resource/Category:",
    "dbd": "http://dbpedia.org/datatype/",
    "dbo": "http://dbpedia.org/ontology/",
    "dbp": "http://dbpedia.org/property/",
    "dbr": "http://dbpedia.org/resource/",
    "dbt": "http://dbpedia.org/resource/Template:",
    "entity": "http://www.wikidata.org/entity/",
    
    # can cause problems
    "parliament": "https://id.parliament.uk/schema/",
    "parl": "https://id.parliament.uk/schema/",
}

URL_TO_PREFIX = {v: k for k, v in PREFIX_TO_URL.items()}

def send_query_to_api(query, api, timeout_limit, num_try):
    response = None
    while num_try > 0 and response == None and not is_query_empty(query):
        try:
            print(f"| Calling API... ", end="", flush=True)
            sparql_response = api.execute_sparql(query, timeout=timeout_limit)
            response = sparql_response.bindings if sparql_response.success else sparql_response.data
                
        except HTTPError as inst:
            if inst.response.status_code == 429:
                retry_after = int(inst.response.headers['retry-after'])
                print(f"| Retry-after: {retry_after} ", end="", flush=True)
                time.sleep(retry_after + 1)
                num_try -= 1
            else:
                print(f"| Exception occured ", end="", flush=True)
                response = "exception: " + str(inst) + "\n" + inst.response.text
        except Timeout:
            response = "timeout"
            print(f"| Response Timeout ", end="", flush=True)
        except Exception as inst:
            print(f"| Exception occured ", end="", flush=True)
            response = "exception: " + str(inst)
    return response if response != None else "exception: too many retry-after"

def is_query_empty(query :str) -> bool:
    query = query.strip()
    return query is None or query == "" or len(query) == 0

def can_add_limit_clause(query :str) -> bool:
    upper_query = query.upper()
    return (not is_query_empty(query) and not re.search(r"\WCOUNT\W", upper_query) and not re.search(r"\WLIMIT\W", upper_query))

def load_dataset(dataset_path: str):
    if dataset_path.endswith((".parquet.gzip", ".parquet")):
        try:
            return pd.read_parquet(dataset_path, engine="fastparquet")
        except:
            return pd.read_parquet(dataset_path)
    elif dataset_path.endswith(".json"):
        return pd.read_json(dataset_path)
    elif dataset_path.endswith(".pkl"):
        return pd.read_pickle(dataset_path)
    raise ValueError(f"The provided dataset format is not taken in charge. Use json, parquet or pickle. Found: {dataset_path}")

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="SPARQL Queries Executor",
                                    description="Execute queries on Wikidata's SPARQL endpoint")
    parser.add_argument('-d', '--dataset', required=True, type=str, help="Path to the dataset parquet file.")
    parser.add_argument('-c', '--column-name', type=str, help="The column where the queries to be executed are.", default="query")
    parser.add_argument('-t', '--timeout', type=int, help="The amount of time for the HTTP client to wait before timeout.", default=60)
    parser.add_argument('-l', "--limit", type=int, help="If the limit is >0 a LIMIT clause will be added to all non COUNT queries.", default=0)
    parser.add_argument('-o', "--output", required=True, type=str, help="Path to the directory the save file will be.")
    parser.add_argument('-sn', "--save-name", required=True, type=str, help="Name of the save file.")

    args = parser.parse_args()

    dataset_path = args.dataset
    answer_limit = args.limit
    timeout_limit = args.timeout
    do_add_limit = answer_limit > 0

    df_dataset = load_dataset(dataset_path)
    
    if not args.column_name in df_dataset.columns:
        raise ValueError(f"The column '{args.column_name}' was not found in the dataset. Columns are {', '.join(df_dataset.columns)}.")
    
    df_dataset['execution'] = df_dataset.apply(lambda x: None, axis=1)
    df_dataset['executed_query'] = df_dataset.apply(lambda x: None, axis=1)

    api = WikidataAPI()
    
    num_processed = 0
    for (i, query) in df_dataset[args.column_name].items():
        print(f"row {str(num_processed)}/{len(df_dataset)} ".ljust(15), end="", flush=True)
        response = None
        
        if is_query_empty(query):
            response = "exception: query is empty"
            print(f"| Query is empty ", end="", flush=True)
        else:
            query = add_relevant_prefixes_to_query(query)
            
            if do_add_limit and can_add_limit_clause(query):
                query += f"\nLIMIT {answer_limit}"
            
            
            response = send_query_to_api(query=query,
                                    api=api,
                                    timeout_limit=timeout_limit,
                                    num_try=3)
        print(f"| done.", flush=True)
        
        df_dataset.at[i, 'execution'] = str(response)
        df_dataset.at[i, 'executed_query'] = query
        num_processed += 1

    output_file_path = os.path.join(args.output, f"{args.save_name}.parquet.gzip")
    try:
        df_dataset.to_parquet(output_file_path, engine="fastparquet", compression="gzip")
    except:
        df_dataset.to_parquet(output_file_path, compression="gzip")
        