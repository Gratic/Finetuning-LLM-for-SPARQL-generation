import sys
from pathlib import Path
sys.path.append(Path("modules").absolute().__str__())

import openai
from datasets import load_dataset
from prompts_template import get_template_for_model
import random
from sft_peft import (
    get_target_data,
    create_empty_results,
    align_data_lengths,
    tokenize_predictions,
    execute_pipeline,
    execute_query,
    create_and_process_dataframe,
    compute_all_metrics,
    calculate_final_metrics,
)
from typing import List, Dict, Any
import evaluate
from tqdm import tqdm
import json
import argparse
from functools import wraps
import time

ELABORATE_INSTRUCTION = """Your assignment involves a two-step process:
First, you will receive a task described in a single sentence. This task outlines what information you need to find or what question you need to answer. Read it properly.
Then, you will create a SPARQL Query: Based on the task given, you will write a SPARQL query. SPARQL is a specialized query language used to retrieve and manipulate data stored in Resource Description Framework (RDF) format. Your goal is to craft a query that, when executed, fetches the data or answers required by the initial instruction.
Make sure your SPARQL query is correctly formulated so that, upon execution, it produces the desired result matching the task's requirements.
[examples]
Answer this following instruction:"""

def retry_after(seconds, max_attempts=3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        raise e
                    print(f"Attempt {attempts} failed. Retrying in {seconds} seconds...")
                    time.sleep(seconds)
        return wrapper
    return decorator

def create_format_prompt(input_col:str, target_col:str, with_target: bool, template:str, start_tag:str, end_tag:str, n_examples:int=0):
    def format_prompt(examples):
        output_texts = []
        label_texts = []
        for i in range(len(examples[input_col])):
            n = len(examples[input_col][i])
            
            # llama 3 - remove system prompt from template
            text = template.replace("<|start_header_id|>system<|end_header_id|>\n\n[system_prompt]<|eot_id|>", "")
            
            prompt = ELABORATE_INSTRUCTION
            
            # Add randomly sampled examples if n_examples > 0
            if n_examples > 0:
                examples_text = "\n\nHere are some examples:\n\n"
                available_indices = list(range(len(examples[input_col])))
                available_indices.remove(i)
                for _ in range(n_examples):
                    random_idx = random.choice(available_indices)
                    available_indices.remove(random_idx)
                    n_2 = len(examples[input_col][random_idx])
                    
                    examples_text += f"Input: {examples[input_col][random_idx][random.randint(0, n_2-1)]}\n"
                    examples_text += f"Output: {start_tag}{examples[target_col][random_idx]}{end_tag}\n\n"
                prompt = prompt.replace('[examples]', examples_text)
            prompt += examples[input_col][i][random.randint(0, n-1)]
            
            text = text.replace('[prompt]', prompt)
            
            answer_text = f'\n{start_tag}{examples[target_col][i]}{end_tag}'
            
            if with_target:
                text += answer_text
        
            output_texts.append(text)
            label_texts.append(answer_text)
            
        return {"text": output_texts}
    return format_prompt

def create_compute_metrics(eval_dataset, target_column, rouge_metric, bleu_metric, meteor_metric, start_tag, end_tag):
    # https://huggingface.co/docs/evaluate/transformers_integrations
    def compute_metrics(generated_texts: List[str]):
        target_queries, raw_target_queries, executed_target_queries = get_target_data(eval_dataset, target_column)
        
        # All generated_texts are empty.
        if all(len(x) == 0 for x in generated_texts):
            return create_empty_results()
        
        generated_texts, target_queries, raw_target_queries, executed_target_queries = align_data_lengths(
            generated_texts, target_queries, raw_target_queries, executed_target_queries
        )
        
        tokenized_preds = tokenize_predictions(generated_texts)
            
        translated_preds = list(map(lambda x: x['output'], execute_pipeline(generated_texts)))
        executed_preds = [execute_query(query, start_tag, end_tag) for query in translated_preds]

        
        data, df_not_null = create_and_process_dataframe(executed_target_queries, executed_preds)
        
        nested_metrics, cross_metrics, id_metrics = compute_all_metrics(data, df_not_null)
        
        results_dict = calculate_final_metrics(
            nested_metrics, cross_metrics, id_metrics, 
            tokenized_preds, target_queries, executed_preds,
            rouge_metric, bleu_metric, meteor_metric
        )
        
        return results_dict
    return compute_metrics

@retry_after(seconds=120, max_attempts=5)
def generate_single_response(client, model, max_tokens, temperature, prompt):
    return client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )

def generate_responses(client: openai.OpenAI, prompts: List[str], model: str, max_tokens: int, temperature: float) -> List[str]:
    responses = []
    for prompt in tqdm(prompts, desc="Generating responses"):
        try:
            response = generate_single_response(client, model, max_tokens, temperature, prompt)
            responses.append(response.choices[0].message.content.strip())
        except Exception as e:
            responses.append("")
    return responses

def parse_arguments() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="OpenAI Evaluation Script")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--base-url", required=True, help="OpenAI API base URL")
    parser.add_argument("--dataset-name", required=True, help="Name of the dataset to use")
    parser.add_argument("--dataset-split", default="valid", help="Dataset split to use (default: valid)")
    parser.add_argument("--input-column", required=True, help="Input column name in the dataset")
    parser.add_argument("--target-column", required=True, help="Target column name in the dataset")
    parser.add_argument("--start-tag", required=True, help="Start tag for the output")
    parser.add_argument("--end-tag", required=True, help="End tag for the output")
    parser.add_argument("--model", default="mistral", choices=["mistral", "llama3"], help="Model to use (default: mistral)")
    parser.add_argument("--max-tokens", type=int, required=True, help="Maximum number of tokens for generation")
    parser.add_argument("--temperature", type=float, required=True, help="Temperature for generation")
    parser.add_argument("--n-examples", type=int, default=3, help="Number of examples to include (default: 3)")
    
    args = parser.parse_args()
    
    # Convert arguments to a dictionary
    config = vars(args)
    
    # Update model_id based on the selected model
    MODEL_TO_ID = {
        "mistral": "mistral",
        "llama3": "llama-3",
    }
    config["model_id"] = MODEL_TO_ID[config["model"]]
    
    return config

def main(config: Dict[str, Any]):
    client = openai.OpenAI(
        api_key=config['api_key'],
        base_url=config['base_url']
    )

    dataset = load_dataset(config['dataset_name'], split=config['dataset_split'])

    formatting_func = create_format_prompt(
        input_col=config['input_column'],
        target_col=config['target_column'],
        with_target=False,
        template=get_template_for_model(config['model_id']),
        start_tag=config['start_tag'],
        end_tag=config['end_tag'],
        n_examples=config['n_examples'],
    )

    dataset = dataset.map(formatting_func, batched=True, load_from_cache_file=False)

    compute_metrics = create_compute_metrics(
        dataset,
        config['target_column'],
        rouge_metric=evaluate.load("rouge"),
        bleu_metric=evaluate.load("bleu"),
        meteor_metric=evaluate.load("meteor"),
        start_tag=config['start_tag'],
        end_tag=config['end_tag'],
    )

    generated_responses = generate_responses(
        client=client,
        prompts=dataset['text'],
        model=config["model"],
        max_tokens=config["max_tokens"],
        temperature=config["temperature"]
        )

    results = compute_metrics(generated_responses)

    output_data = {
        "prompts": dataset['text'],
        "generated_responses": generated_responses,
        "metrics": results
    }

    with open(f"evaluation_results_{config['model']}_{config['input_column']}_{config['target_column']}.json", "w") as f:
        json.dump(output_data, f, indent=2)

if __name__ == "__main__":
    config = parse_arguments()
    main(config)