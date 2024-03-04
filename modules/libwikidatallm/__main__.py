import sys
from pathlib import Path
sys.path.append(Path("modules").absolute().__str__())

from .EntityExtractor import LLMEntityExtractor, BracketRegexEntityExtractor
from .EntityLinker import TakeFirstWikidataEntityLinker
from .LLMConnector import LlamaCPPConnector, vLLMConnector, PeftConnector, LLMConnector
from .Pipeline import OrderedPipeline
from .PipelineFeeder import SimplePipelineFeeder
from .PlaceholderFiller import SimplePlaceholderFiller
from .TemplateLLMQuerySender import TemplateLLMQuerySender, BASE_LLAMA_TEMPLATE, BASE_MISTRAL_TEMPLATE
from .Translator import LLMTranslator, BASE_ANNOTATED_INSTRUCTION, BASE_ANNOTATED_INSTRUCTION_ONE_SHOT
from .SentencePlaceholder import SimpleSentencePlaceholder
import pandas as pd
import os
import argparse
from data_utils import set_seed, load_dataset

def basic_pipeline(dataset: pd.DataFrame, column: str, llm_connector: LLMConnector, template: str = BASE_MISTRAL_TEMPLATE, use_tqdm: bool = False):
    pipeline = OrderedPipeline()
    
    templateLLMQuerySender = TemplateLLMQuerySender(llm_connector, template, "[", "]")
    pipeline.add_step(LLMTranslator(templateLLMQuerySender, "", BASE_ANNOTATED_INSTRUCTION))
    
    feeder = SimplePipelineFeeder(pipeline, use_tqdm=use_tqdm)
    return feeder.process(dataset[column])

def template_pipeline(dataset: pd.DataFrame, column: str, llm_connector: LLMConnector, template: str = BASE_MISTRAL_TEMPLATE, use_tqdm: bool = False):
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
        instruction_prompt=BASE_ANNOTATED_INSTRUCTION,
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
        output_column='linked_query'
        )
    
    pipeline.add_step(translator)
    pipeline.add_step(entity_extractor)
    pipeline.add_step(entity_linker)
    pipeline.add_step(query_filler)
    
    feeder = SimplePipelineFeeder(pipeline, use_tqdm=use_tqdm)
    return feeder.process(dataset[column])

def execute_pipeline(args: argparse.Namespace, dataset: pd.DataFrame, llm_connector: LLMConnector, use_tqdm: bool=False):
    if not args.pipeline in ['basic', 'template']:
        raise ValueError(f"Please between 'basic' and 'template', found: {args.pipeline}.")
    
    template = BASE_LLAMA_TEMPLATE if args.model.lower().contains("llama") else BASE_MISTRAL_TEMPLATE
    
    if args.pipeline == "basic":
        return basic_pipeline(dataset, args.column_name, llm_connector, template, use_tqdm)
    if args.pipeline == "template":
        return template_pipeline(dataset, args.column_name, llm_connector, template, use_tqdm)

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
            temperature=args.temperature,
            top_p=args.topp,
            max_number_of_tokens_to_generate=args.num_tokens
        )
    raise ValueError(f"The only engines supported is 'vllm' and 'peft', found: {args.engine}.")

def input_normalization(args: argparse.Namespace, dataset: pd.DataFrame):
    dataset[args.column_name] = dataset.apply(lambda x: x[args.column_name][0] if isinstance(x[args.column_name], list) else x[args.column_name], axis=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="LLM Inference pipeline SparQL",
                                     description="Script to generate SPARQL queries using LLM.")
    parser.add_argument("-d", "--data", type=str, help="Path to the pickle train dataset.")
    parser.add_argument("-c", "--column-name", type=str, help="The column name to use as input. For non-interactive mode only. Default='input'", default='input')
    parser.add_argument("-m", "--model", required=True, type=str, help="Path to the model.")
    parser.add_argument("-a", "--adapters", type=str, help="Path to the adapter models.")
    parser.add_argument("-tok", "--tokenizer", required=True, type=str, help="Path to the tokenizer.")
    parser.add_argument("-ctx", "--context-length", type=int, help="Maximum context length of the LLM.", default=2048)
    parser.add_argument("-e", "--engine", type=str, help="Which engine to use (vllm only right now).", default="vllm", choices=["vllm", "peft"])
    parser.add_argument("-pl", "--pipeline", type=str, help="Which pipeline to use (basic and template).", default="basic", choices=["basic", "template"])
    parser.add_argument("-t", "--temperature", type=float, help="Temperature for decoder.", default=0.2)
    parser.add_argument("-topp", "--topp", type=float, help="Top-p for decoder.", default=0.95)
    parser.add_argument("-ntok", "--num-tokens", type=int, help="Maximum number of tokens generated.", default=256)
    parser.add_argument("-i", "--interactive", action="store_true", help="Allow the user to input string interactively.")
    parser.add_argument("-tqdm", "--tqdm", action="store_true", help="Use tqdm as a progress bar.")
    parser.add_argument("-o", "--output", type=str, help="Path to the directory to save the file.")
    parser.add_argument("-sn", "--save-name", type=str, help="Name of the file to be save.")
    parser.add_argument("-rand", "--random-seed", type=int, help="Set up a random seed if specified.", default=0)
    
    args = parser.parse_args()
    
    llm_connector = get_llm_engine(args)
    
    if args.random_seed != 0:
        set_seed(args.random_seed)
    
    if args.interactive:
        print("Interactive mode ON: Quit by typing 'q'.")
        
        user_prompt = input("Enter your prompt:")
        while user_prompt != "q":
            dataset = pd.DataFrame(data={args.column_name: [user_prompt]})
            result = execute_pipeline(args, dataset, llm_connector, args.tqdm)[0]
            
            if result['has_error']:
                print("An error has occured.")
            else:
                print(result['translated_prompt'])
            user_prompt = input("Enter your prompt:")
            
        exit(0)
    else:
        if args.data == None or args.data == "":
            raise ValueError("The data argument is required.")
        
        if args.output == None or args.output == "":
            raise ValueError("The output argument is required.")
        
        if args.save_name == None or args.save_name == "":
            raise ValueError("The save-name argument is required.")
        
        if not os.path.exists(args.data):
            raise FileNotFoundError(f"The dataset has not been found: {args.data}")
        
        dataset = load_dataset(args.data)
        
        if args.column_name not in dataset.columns:
            raise ValueError(f"The column {args.column_name} is not present in the dataset columns: {dataset.columns}.")
        
        input_normalization(args, dataset)
            
        results = execute_pipeline(args, dataset, llm_connector, args.tqdm)
        
        os.makedirs(f"{args.output}", exist_ok=True)
        df_export = pd.DataFrame.from_dict(results)
        df_export = df_export.set_index(dataset.index)
        df_export = pd.concat([df_export, dataset], axis=1)
        df_export.to_parquet(os.path.join(args.output, f"{args.save_name}.parquet.gzip"), engine="fastparquet", compression="gzip")