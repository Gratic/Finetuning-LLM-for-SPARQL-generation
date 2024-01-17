import pandas as pd
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import single_meteor_score
import argparse
import os

def failed_generation_index(dataset: pd.DataFrame):
    return dataset.loc[dataset['has_error'] == True].index

def corpus_meteor(references, hypotheses):
    meteor_scores = 0.
    for ref, hyp in zip(references, hypotheses):
        meteor_scores += single_meteor_score(ref, hyp)
    return meteor_scores / float(len(references))

parser = argparse.ArgumentParser(prog="Evaluation bench for LLM",
                                 description="Evaluate LLMs for SPARQL generation")
parser.add_argument('-d', '--dataset', required=True, type=str, help="The path to the dataset.")
parser.add_argument('-m', '--model', required=True, type=str, help="The model name.")
parser.add_argument('-o', '--output', required=True, type=str, help="Folder to output the results.")
parser.add_argument('-sn', '--save-name', required=True, type=str, help="Name of the save file.")

args = parser.parse_args()

df = pd.read_parquet(args.dataset)
df_no_fail = df.drop(failed_generation_index(df))

bleu_score = corpus_bleu([[x.split()] for x in df_no_fail['target']], [x.split() for x in df_no_fail['translated_prompt']])
meteor_score = corpus_meteor(df_no_fail['target'], df_no_fail['translated_prompt'])
serie = pd.Series(data=
                  {
                      "model_name": args.model,
                      "num_rows": len(df),
                      "num_fail": len(df.loc[df['has_error'] == True]),
                      "bleu_score": bleu_score,
                      "meteor_score": meteor_score,
                  })

serie.to_json(os.path.join([args.output, f"{args.save_name}.json"]))