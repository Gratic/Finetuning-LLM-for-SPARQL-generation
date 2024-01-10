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
    
    print(f"{df_output.iloc[[0]]=}")
    print(f"{df_output.iloc[[0]]['input']=}")
    print(f"{df_output.iloc[[0]]['target']=}")
    
    # Shuffling
    print("Shuffling...")
    df_output = df_output.sample(frac=1).reset_index(drop=True)
    
    print(f"{df_output.iloc[[0]]=}")
    print(f"{df_output.iloc[[0]]['input']=}")
    print(f"{df_output.iloc[[0]]['target']=}")
    
    # Splitting
    split_index = int(0.8 * len(df_output))
    valid_index = split_index + int(0.25 * (len(df_output) - split_index))
    print(f"{split_index=}")
    print(f"{valid_index=}")
    
    df_train = df_output.iloc[:split_index]
    df_valid = df_output.iloc[split_index:valid_index]
    df_test = df_output.iloc[valid_index:]
    
    print(f"{len(df_output)=}")
    print(f"{len(df_train)=}")
    print(f"{len(df_valid)=}")
    print(f"{len(df_test)=}")
    
    train_file = arguments.output + "finetune_dataset_train.pkl"
    print(f"train dataset saved at: {train_file}") 
    df_train.to_pickle(train_file)
    
    valid_file = arguments.output + "finetune_dataset_valid.pkl"
    print(f"valid dataset saved at: {valid_file}")
    df_valid.to_pickle(valid_file)
    
    test_file = arguments.output + "finetune_dataset_test.pkl"
    print(f"test dataset saved at: {test_file}")
    df_test.to_pickle(test_file)