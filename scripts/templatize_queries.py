import sys
from pathlib import Path
sys.path.append(Path("modules").absolute().__str__())

import argparse
from data_utils import load_dataset
from execution_utils import add_relevant_prefixes_to_query
from libwikidatallm.EntityFinder import WikidataAPI
import re
from tqdm import tqdm

def extract_entities_properties_ids(query:str):
    pattern = re.compile(r":(Q\d+|P\d+)")
    results = pattern.findall(query)

    if results:
        return results
    else:
        return []

def replace_entities_and_properties_id_with_labels(query: str):
    extracted_properties_and_entities = set(extract_entities_properties_ids(query))
    
    api = WikidataAPI()
    
    entities_id_w_labels = [api._smart_get_labels_from_entity_id(entity_id)[0] for entity_id in filter(lambda x: x.startswith("Q"), extracted_properties_and_entities)]
    properties_id_w_labels = [api._smart_get_labels_from_entity_id(property_id)[0] for property_id in filter(lambda x: x.startswith("P"), extracted_properties_and_entities)]
    
    new_query = query
    for e, label in entities_id_w_labels:
        new_query = re.sub(e, f"[entity:{label}]", new_query)
    for p, label in properties_id_w_labels:
        new_query = re.sub(p, f"[property:{label}]", new_query)
    
    return new_query
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Templatize queries", description="Takes the queries and templatize them.")
    parser.add_argument("-i", "--input", help="The dataset to use.", required=True)
    parser.add_argument("-c", "--column", help="The column the queries are in. Default='query'", default='query')
    parser.add_argument("-oc", "--out-column", help="The column to put the templated queries. Default='query_templated'", default='query_templated')
    parser.add_argument("-o", "--output", help="The folder to output to.", required=True)
    parser.add_argument("-sn", "--save-name", help="Name of the file to output", required=True)
    parser.add_argument("-p", "--prefix", action="store_true", help="Add the relevant SPARQL prefix to the query.")
    
    args = parser.parse_args()
    
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    file_to_output = output / f"{args.save_name}.json"
        
    dataset = load_dataset(args.input)
    
    if args.column not in dataset.columns:
        raise ValueError(f"There is no column {args.column} in the dataset, there are: {dataset.columns}.")
    
    tqdm.pandas()
    dataset[args.out_column] = dataset.progress_apply(lambda x: replace_entities_and_properties_id_with_labels(x[args.column]) if not args.prefix else replace_entities_and_properties_id_with_labels(add_relevant_prefixes_to_query(x[args.column])), axis=1)
        
    dataset.to_json(file_to_output)