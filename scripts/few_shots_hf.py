import sys
from pathlib import Path
sys.path.append(Path("modules").absolute().__str__())

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
import torch

# New imports for Transformers, PEFT, and bitsandbytes
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel, PeftConfig

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
            
            text = template.replace("<|start_header_id|>system<|end_header_id|>\n\n[system_prompt]<|eot_id|>", "")
            text = template.replace("[system_prompt]", "")
            
            prompt = ELABORATE_INSTRUCTION
            
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
            else:
                prompt = prompt.replace('[examples]', '')
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
    def compute_metrics(generated_texts: List[str]):
        target_queries, raw_target_queries, executed_target_queries = get_target_data(eval_dataset, target_column)
        
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
def generate_single_response(model, tokenizer, generation_config, prompt):
    model.eval()
    with torch.inference_mode():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, generation_config=generation_config, tokenizer=tokenizer)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]

def generate_responses(model, tokenizer, prompts: List[str], generation_config: GenerationConfig) -> List[str]:
    responses = []
    for prompt in tqdm(prompts, desc="Generating responses"):
        try:
            response = generate_single_response(model, tokenizer, generation_config, prompt)
            responses.append(response.strip())
        except Exception as e:
            responses.append("")
    return responses

def parse_arguments() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Hugging Face Transformers Evaluation Script")
    parser.add_argument("--dataset-name", required=True, help="Name of the dataset to use")
    parser.add_argument("--dataset-split", default="valid", help="Dataset split to use (default: valid)")
    parser.add_argument("--input-column", required=True, help="Input column name in the dataset")
    parser.add_argument("--target-column", required=True, help="Target column name in the dataset")
    parser.add_argument("--start-tag", required=True, help="Start tag for the output")
    parser.add_argument("--end-tag", required=True, help="End tag for the output")
    parser.add_argument("--base-model", required=True, help="Base model name from Hugging Face")
    parser.add_argument("--lora-path", required=True, help="Path to the LoRA adapter")
    parser.add_argument("--max-tokens", type=int, required=True, help="Maximum number of tokens for generation")
    parser.add_argument("--temperature", type=float, required=True, help="Temperature for generation")
    parser.add_argument("--n-examples", type=int, default=3, help="Number of examples to include (default: 3)")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for inference if available")
    parser.add_argument("--quantization", choices=["4bit", "8bit", "no"], default="no", help="Quantization level (default: no)")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32", help="Model precision (default: fp32)")
    parser.add_argument("--output", default=".", help="Folder to save the output (default: current directory)")
    parser.add_argument("--save-name", default="evaluation_results", help="Name of the saved file without extension (default: evaluation_results)")
    
    args = parser.parse_args()
    
    return vars(args)

def main(config: Dict[str, Any]):
    torch_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16
    }[config['precision']]
    
    if config['quantization'] == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype
            )
    elif config['quantization'] == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quantization_config = None
    
    temperature = config["temperature"]

    device = "cuda:0" if config['use_gpu'] and torch.cuda.is_available() else "cpu"

    base_model = AutoModelForCausalLM.from_pretrained(
        config['base_model'],
        quantization_config=quantization_config,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch_dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
    tokenizer.padding_side = "left"
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = PeftModel.from_pretrained(base_model, config['lora_path'])

    model = model.to(device)

    generation_config = GenerationConfig(
        max_new_tokens=config["max_tokens"],
        temperature=None if temperature == 0 else temperature,
        do_sample=temperature > 0,
        use_cache=True,  # Enable caching of key/value pairs
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        stop_strings=config['end_tag'],
        tokenizer=tokenizer,
    )

    dataset = load_dataset(config['dataset_name'], split=config['dataset_split'])

    formatting_func = create_format_prompt(
        input_col=config['input_column'],
        target_col=config['target_column'],
        with_target=False,
        template=get_template_for_model(config['base_model']),
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
        model=model,
        tokenizer=tokenizer,
        prompts=dataset['text'],
        generation_config=generation_config
    )

    results = compute_metrics(generated_responses)

    output_data = {
        "prompts": dataset['text'],
        "generated_responses": generated_responses,
        "metrics": results
    }

    output_path = Path(config['output'])
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{config['save_name']}.json"

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    config = parse_arguments()
    main(config)