import pandas as pd
import argparse
import re

class MySPARQL():
    def __init__(self, raw_query: str) -> None:
        self.raw_query = raw_query
        self.ent_to_var, self.var_to_ent = self._gen_dict()
        self.template_form = self._templatize()

    def _gen_dict(self):
        matches = re.findall(r"\w+:\w+", self.raw_query)
        
        unique = []
        for x in matches:
            if x not in unique:
                unique.append(x)        
        
        ent_to_var = {match:f"[entity {i}]" for i,match in enumerate(unique)}
        var_to_ent = {value:key for key,value in ent_to_var.items()}
        
        return ent_to_var, var_to_ent
    
    def _templatize(self):
        template_form = self.raw_query
        for key, value in self.ent_to_var.items():
            template_form = template_form.replace(key, value)
        return template_form

def transform(raw_query):
    sparql = MySPARQL(raw_query)
    return sparql.template_form + "\n\n" + "\n".join([f"{key}: {value}" for key, value in sparql.var_to_ent.items()])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Path to the pandas dataset with result column and query column.", required=True)
    parser.add_argument("-o", "--output", type=str, help="Path to output directory.", default="./outputs/")

    arguments = parser.parse_args()

    with open(arguments.input, "r") as f:
        data = f.read()

    df = pd.read_json(data)

    df['input'] = df.apply(lambda x: x['result'], axis=1)
    df['target'] = df.apply(lambda x: transform(x['query']), axis=1)
    
    df_output = df[['input', 'target']]
    
    df_output.to_pickle(f"{arguments.output}finetune_dataset.pkl")