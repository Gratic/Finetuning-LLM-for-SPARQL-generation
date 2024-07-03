import sys
from pathlib import Path
sys.path.append(Path("modules").absolute().__str__())

from .EntityExtractor import LLMEntityExtractor, BracketRegexEntityExtractor
from .EntityLinker import TakeFirstWikidataEntityLinker
from .LLMConnector import LlamaCPPConnector, vLLMConnector, PeftConnector, LLMConnector
from .Pipeline import OrderedPipeline
from .PipelineFeeder import SimplePipelineFeeder
from .PlaceholderFiller import SimplePlaceholderFiller
from .SentencePlaceholder import SimpleSentencePlaceholder
from .TemplateLLMQuerySender import TemplateLLMQuerySender
from .Translator import LLMTranslator
from data_utils import set_seed, load_dataset
from prompts_template import BASE_BASIC_INSTRUCTION, LLAMA2_TEMPLATE, BASE_MISTRAL_TEMPLATE, ELABORATE_INSTRUCTION, get_template_for_model
from execution_utils import prepare_and_send_query_to_api
from huggingface_hub import login
from typing import Dict, List
import argparse
import os
import pandas as pd
import time
import datasets

def basic_pipeline(llm_connector: LLMConnector, template: str = BASE_MISTRAL_TEMPLATE, start_tag:str="[query]", end_tag:str="[/query]"):
    pipeline = OrderedPipeline()
    
    templateLLMQuerySender = TemplateLLMQuerySender(llm_connector, template, "[", "]")
    pipeline.add_step(LLMTranslator(
        templateQuerySender=templateLLMQuerySender, 
        system_prompt="", 
        instruction_prompt=ELABORATE_INSTRUCTION,
        start_tag=start_tag,
        end_tag=end_tag,
        input_column="row",
        output_column="output"
        ))
    
    return pipeline
    
def template_pipeline(llm_connector: LLMConnector, template: str = BASE_MISTRAL_TEMPLATE, start_tag:str="[query]", end_tag:str="[/query]"):
    pipeline = OrderedPipeline()

    # 1. Generate answer, answer will be templated (a llm trained that way is needed)
    templateLLMQuerySender = TemplateLLMQuerySender(
        llm=llm_connector,
        template_text=template,
        start_seq='[',
        end_seq=']'
        )
    translator = LLMTranslator(
        templateQuerySender=templateLLMQuerySender,
        system_prompt='',
        instruction_prompt=ELABORATE_INSTRUCTION,
        start_tag=start_tag,
        end_tag=end_tag,
        input_column='row',
        output_column='translated_prompt'
        )
    
    # 2. From the templated answer extract the values
    entity_extractor = BracketRegexEntityExtractor(
        input_column=translator.output_column,
        output_col_entities='extracted_entities',
        output_col_properties='extracted_properties'
        )
    
    # 3. Reverse search the closest values from labels (a possible upgrade for later: Use an LLM to choose the best values)
    entity_linker = TakeFirstWikidataEntityLinker(
        input_column_entities=entity_extractor.output_col_entities,
        input_column_properties=entity_extractor.output_col_properties,
        output_column_entities='linked_entities',
        output_column_properties='linked_properties'
        )
    
    # 4. Replace the labels with the ID we found in 3.
    query_filler = SimplePlaceholderFiller(
        input_column_query=translator.output_column,
        input_column_entities=entity_linker.output_column_entities,
        input_column_properties=entity_linker.output_column_properties,
        output_column='output'
        )
    
    pipeline.add_step(translator)
    pipeline.add_step(entity_extractor)
    pipeline.add_step(entity_linker)
    pipeline.add_step(query_filler)
    
    return pipeline

def execute_pipeline(args: argparse.Namespace, dataset: pd.DataFrame, llm_connector: LLMConnector, use_tqdm: bool=False) -> List[Dict]:
    if not args.pipeline in ['basic', 'template']:
        raise ValueError(f"Please between 'basic' and 'template', found: {args.pipeline}.")
    
    template = get_template_for_model(args.model)
    
    if args.pipeline == "basic":
        pipeline = basic_pipeline(llm_connector, template, start_tag=args.start_tag, end_tag=args.end_tag)
    if args.pipeline == "template":
        pipeline = template_pipeline(llm_connector, template, start_tag=args.start_tag, end_tag=args.end_tag)

    feeder = SimplePipelineFeeder(pipeline, use_tqdm=use_tqdm)
    results = feeder.process(dataset[args.column_name])
    
    # Make sure there is an output column in the dataset...
    list(map(lambda x: x.update({"output": None}),
        filter(lambda x: 'output' not in list(x.keys()),
               results)))
    
    return results

def get_llm_engine(args):
    if args.engine == "vllm":
        return vLLMConnector(model_path=args.model,
                        tokenizer=args.tokenizer,
                        context_length=args.context_length,
                        temperature=args.temperature,
                        top_p=args.topp,
                        max_number_of_tokens_to_generate=args.num_tokens)
    elif args.engine == "peft":
        return PeftConnector(
            model_path=args.model,
            adapter_path=args.adapters,
            context_length=args.context_length,
            dtype=args.computational_type,
            decoding_strategy=args.decoding,
            temperature=args.temperature,
            top_p=args.topp,
            max_number_of_tokens_to_generate=args.num_tokens,
        )
    raise ValueError(f"The only engines supported is 'vllm' and 'peft', found: {args.engine}.")

def input_normalization(args: argparse.Namespace, dataset: pd.DataFrame):
    dataset[args.column_name] = dataset.apply(lambda x: x[args.column_name][0] if isinstance(x[args.column_name], list) else x[args.column_name], axis=1)

def get_args(list_args=None):
    parser = argparse.ArgumentParser(prog="LLM Inference pipeline SparQL",
                                     description="Script to generate SPARQL queries using LLM.")
    parser.add_argument("-d", "--data", type=str, help="Path to the pickle train dataset.")
    parser.add_argument("-hf", "--huggingface-dataset", type=str, help="Path to the huggingface dataset.")
    parser.add_argument("-hfs", "--huggingface-split", type=str, help="Split to use from the huggingface dataset.", default="train")
    parser.add_argument("-c", "--column-name", type=str, help="The column name to use as input. For non-interactive mode only. Default='input'", default='input')
    parser.add_argument("-m", "--model", required=True, type=str, help="Path to the model.")
    parser.add_argument("-a", "--adapters", type=str, help="Path to the adapter models.")
    parser.add_argument("-tok", "--tokenizer", required=True, type=str, help="Path to the tokenizer.")
    parser.add_argument("-ctx", "--context-length", type=int, help="Maximum context length of the LLM.", default=2048)
    parser.add_argument("-e", "--engine", type=str, help="Which engine to use (vllm and peft).", default="vllm", choices=["vllm", "peft"])
    parser.add_argument("-cd", "--computational-type", type=str, help="Which type the model should be converted to. Choices are 'fp32', 'fp16', and 'bf16'. Default is fp32. Only work for PEFT engine.", default="fp32", choices=['fp32', 'fp16', 'bf16'])
    parser.add_argument("-pl", "--pipeline", type=str, help="Which pipeline to use (basic and template).", default="basic", choices=["basic", "template"])
    parser.add_argument("-st", "--start-tag", type=str, help="Opening tag to search for the query in the LLM response.", default="[query]")
    parser.add_argument("-et", "--end-tag", type=str, help="Closing tag to search for the query in the LLM response.", default="[/query]")
    parser.add_argument("-de", "--decoding", type=str, help="The decoding strategy to use. Can be 'sampling' or 'greedy'. Works only with peft engine.", default="sampling", choices=['sampling', 'greedy'])
    parser.add_argument("-t", "--temperature", type=float, help="Temperature for decoder.", default=0.2)
    parser.add_argument("-topp", "--topp", type=float, help="Top-p for decoder.", default=0.95)
    parser.add_argument("-ntok", "--num-tokens", type=int, help="Maximum number of tokens generated.", default=256)
    parser.add_argument("-i", "--interactive", action="store_true", help="Allow the user to input string interactively.")
    parser.add_argument("-tqdm", "--tqdm", action="store_true", help="Use tqdm as a progress bar.")
    parser.add_argument("-o", "--output", type=str, help="Path to the directory to save the file.")
    parser.add_argument("-sn", "--save-name", type=str, help="Name of the file to be save.")
    parser.add_argument("-rand", "--random-seed", type=int, help="Set up a random seed if specified.", default=0)
    parser.add_argument("-at", "--token", type=str, help="Auth token for gated models (like LLaMa 2).", default="")
    
    args = parser.parse_args(list_args)
    
    args.start_tag = args.start_tag.replace("\\n", "\n")
    args.end_tag = args.end_tag.replace("\\n", "\n")
    return args

def load_data(args):
    if args.huggingface_dataset:
        return datasets.load_dataset(args.huggingface_dataset, split=args.huggingface_split)
    return load_dataset(args.data)

def validate_args_and_data(args):
    if not args.huggingface_dataset and not args.data:
        raise ValueError("Either 'data' or 'huggingface_dataset' argument is required.")
    if not args.output:
        raise ValueError("The 'output' argument is required.")
    if not args.save_name:
        raise ValueError("The 'save_name' argument is required.")
    if not args.huggingface_dataset and not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    dataset = load_data(args)
    
    if args.column_name not in dataset.columns:
        raise ValueError(f"Column '{args.column_name}' not found in dataset. Available columns: {dataset.columns}")
    
    return dataset

def process_data(args, dataset, llm_connector):
    input_normalization(args, dataset)
    return execute_pipeline(args, dataset, llm_connector, args.tqdm)

def save_results(args, results, dataset):
    os.makedirs(args.output, exist_ok=True)
    df_export = pd.DataFrame.from_dict(results).set_index(dataset.index)
    df_export = pd.concat([df_export, dataset], axis=1)
    output_path = os.path.join(args.output, f"{args.save_name}.parquet.gzip")
    df_export.to_parquet(output_path, engine="fastparquet", compression="gzip")

if __name__ == "__main__":
    args = get_args()
    
    if args.token != "":
        login(token=args.token)
    llm_connector = get_llm_engine(args)
    
    if args.random_seed != 0:
        set_seed(args.random_seed)
    
    if args.interactive:
        print("Interactive mode ON: Quit by typing 'q'.")
        
        user_prompt = input("Enter your prompt:")
        while user_prompt != "q":
            start = time.process_time_ns()
            dataset = pd.DataFrame(data={args.column_name: [user_prompt]})
            result = execute_pipeline(args, dataset, llm_connector, args.tqdm)[0]
            end = time.process_time_ns()
            
            time_in_sec = (end - start) / 1e9
            
            print("Generation took: {time_in_sec}s")
            
            
            if result['has_error']:
                print("An error has occured.")
                print(result['status'])
            else:
                query = result['output']
                print(query)
                
                if args.pipeline == 'template':
                    print("---")
                    print(result['translated_prompt'])
                
                _, response = prepare_and_send_query_to_api(query, do_print=False)
                print("---")
                print(response)
            user_prompt = input("Enter your prompt:")
            
        exit(0)
    else:
        dataset = validate_args_and_data(args)
        results = process_data(args, dataset, llm_connector)
        save_results(args, results, dataset)