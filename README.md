# Fine-Tuning LLMs for SPARQL: Leveraging Synthetic Question-Query Pairs

This repository contains the code used for the master thesis **Fine-Tuning LLMs for SPARQL: Leveraging Synthetic Question-Query Pairs**. It includes:
- Dataset preprocessing (data/analysis_of_cleaned_queries.ipynb)
- Fine-tuning Dataset creation (scripts/final_queries_to_finetuning_dataset.py)
- Huggingface Dataset creation (data/create_hf_dataset.ipynb)
- Model training (scripts/training_to_eval_pipeline.py)
- Analysis Python Notebook (in the data folder)

## Repository organisation

The repository is divided into multiple folders:
- data: notebooks related to modifying, creating or analysing data
- modules: python code developed for this project and reused in multiple parts of the repository
    - libsparqltotext is a module to generate natural instruction from SPARQL queries
    - libwikidatallm is a module to generate queries from a finetuned model
- scripts: python code developed for this project
- tests: unit testing of some modules and scripts

## Installation

The code requires `Python >= 3.8` to work properly.

You can install all the dependencies from the `requirements.txt` file using pip.

```python
pip install -r requirements.txt
```

## Access to the fine-tuning datasets

Creating the fine-tuning dataset is a long process. The fine-tuning datasets used during the thesis are available on Huggingface. There are 4 versions. Two datasets have their natural instruction generated with Mistral Instruct v0.2 Q5_K_M while the others two with Llama 3 70B. Then there is a variant with answers truncated to a maximum of 10 while the other is not truncated.

**Mistral Instruct 7B v0.2 Natural Instructions**
- [Limit 10 (Zaleks/labelized_sparql_wikidata_mistral_v02_q5_limit_10)](https://huggingface.co/datasets/Zaleks/labelized_sparql_wikidata_mistral_v02_q5_limit_10)
- [No Limit (Zaleks/labelized_sparql_wikidata_mistral_v02_q5_no_limit)](https://huggingface.co/datasets/Zaleks/labelized_sparql_wikidata_mistral_v02_q5_no_limit)

**Llama 3 70B Natural Instructions**
- [Limit 10 (Zaleks/labelized_sparql_wikidata_llama3_70B_limit_10)](https://huggingface.co/datasets/Zaleks/labelized_sparql_wikidata_llama3_70B_limit_10)
- [No Limit (Zaleks/labelized_sparql_wikidata_llama3_70B_no_limit)](https://huggingface.co/datasets/Zaleks/labelized_sparql_wikidata_llama3_70B_no_limit)

## Usage

Two scripts are orchestrate the two big pipelines. One to start from the base dataset and generate the finetuning dataset and the other for fine-tuning and evaluating a model.

### Generating a fine-tuning dataset

If you want to generate a fine-tuning dataset you can use [final_queries_to_finetuning_dataset.py](.\scripts\final_queries_to_finetuning_dataset.py). A template config file is given in [config_dataset_template.ini](./config_dataset_template.ini).

This scripts does:
- Labelization (often still referenced Templatization): the queries are duplicated into a new column and their entities and properties are replaced with their labels.
- Natural Instructions generation: the queries (both labeled and not) are sent to a LLM that is tasked to generated what a user could prompt to generate the given query.
- Query Execution: the queries are executed on Wikidata's SPARQL Endpoint.
- Dataset Spliting: the queries are then randomly splitted into 3 splits.

To launch the script use this command. `id` will serves as the output folder where everything will be stored during the processing.

```shell
python scripts/final_queries_to_finetuning_dataset.py \
    --config "Your Config File Here" \
    --id "The Name Of The Run"
```

You can modify behavior using those arguments:
- `--output`: change the output folder (where the id folder containing all intermediate and final result will be outputted)
- `--continue-execution`: if the execution terminated abruptly which can happen, recover the process and start from where it was
- `--re-execute-errors`: try executing only queries that failed once more

Important Output files:
- `generated_prompt-executed.parquet.gzip`: this file contains all the queries and their execution
- The splits:
    - `ID-split_train.pkl`: training split (75% of the total "working" dataset)
    - `ID-split_valid.pkl`: validation split (5%)
    - `ID-split_test.pkl`: testing split (20%)

### Fine-tuning models and evaluation on the test split

If you want to finetune models you can use [training_to_eval_pipeline.py](.\scripts\training_to_eval_pipeline.py). A template config file is given in [config_orchestration_template.ini](./config_orchestration_template.ini).

This script does:
- Supervised Fine-Tuning (SFT)
- Generate queries
- Execute the queries
- Evaluate the queries

To launch the script use this command.

```bash
python scripts/training_to_eval_pipeline.py \
    --config "Your Config File" \
    --id "Your project name"
```

Additional options are available:
- `--output`: change the output folder
- `--recover`: continue where the script has been stopped, it does not continue training of a model that has been started and stopped
- `--training-only`: only trains the model without the evaluation part

