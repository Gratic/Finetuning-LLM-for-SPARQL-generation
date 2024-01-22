from .EntityExtractor import LLMEntityExtractor
from .EntityLinker import FirstWikidataEntityLinker
from .LLMConnector import LlamaCPPConnector, vLLMConnector
from .Pipeline import OrderedPipeline
from .PipelineFeeder import SimplePipelineFeeder
from .PlaceholderFiller import SimplePlaceholderFiller
from .TemplateLLMQuerySender import TemplateLLMQuerySender, BASE_LLAMA_TEMPLATE, BASE_MISTRAL_TEMPLATE
from .Translator import LLMTranslator, BASE_ANNOTATED_INSTRUCTION, BASE_ANNOTATED_INSTRUCTION_ONE_SHOT
from .SentencePlaceholder import SimpleSentencePlaceholder
import pandas as pd
import os
import argparse

def basic_pipeline(dataset, llm_connector):
    pipeline = OrderedPipeline()
    
    templateLLMQuerySender = TemplateLLMQuerySender(llm_connector, BASE_MISTRAL_TEMPLATE, "[", "]")
    pipeline.add_step(LLMTranslator(templateLLMQuerySender, "", BASE_ANNOTATED_INSTRUCTION))
    
    feeder = SimplePipelineFeeder(pipeline)
    return feeder.process(dataset['input'])

def template_pipeline(dataset, llm_connector):
    pipeline = OrderedPipeline()

    templateLLMQuerySender = TemplateLLMQuerySender(llm_connector, BASE_LLAMA_TEMPLATE, '[', ']')
    pipeline.add_step(LLMEntityExtractor(templateLLMQuerySender))
    pipeline.add_step(SimpleSentencePlaceholder())
    pipeline.add_step(FirstWikidataEntityLinker())
    pipeline.add_step(LLMTranslator(templateLLMQuerySender))
    pipeline.add_step(SimplePlaceholderFiller())
    
    feeder = SimplePipelineFeeder(pipeline)
    return feeder.process(dataset['input'])

def execute_pipeline(pipeline, dataset, llm_connector):
    if not pipeline in ['basic', 'template']:
        raise ValueError(f"Please between 'basic' and 'template', found: {pipeline}.")
    
    if pipeline == "basic":
        return basic_pipeline(dataset, llm_connector)
    if pipeline == "template":
        return template_pipeline(dataset, llm_connector)

def get_llm_engine(args):
    if args.engine == "vllm":
        return vLLMConnector(model_path=args.model,
                        tokenizer=args.tokenizer,
                        context_length=args.context_length,
                        temperature=args.temperature,
                        top_p=args.topp,
                        max_number_of_tokens_to_generate=args.num_tokens)
    raise ValueError(f"The only engine supported is vllm, found: {args.engine}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="LLM Inference pipeline SparQL",
                                     description="Script to generate SPARQL queries using LLM.")
    parser.add_argument("-td", "--test-data", required=True, type=str, help="Path to the pickle train dataset.")
    parser.add_argument("-m", "--model", required=True, type=str, help="Path to the model.")
    parser.add_argument("-tok", "--tokenizer", required=True, type=str, help="Path to the tokenizer.")
    parser.add_argument("-ctx", "--context-length", type=int, help="Maximum context length of the LLM.", default=2048)
    parser.add_argument("-e", "--engine", type=str, help="Which engine to use (vllm only right now).", default="vllm", choices=["vllm"])
    parser.add_argument("-pl", "--pipeline", type=str, help="Which pipeline to use (basic and template).", default="basic", choices=["basic", "template"])
    parser.add_argument("-t", "--temperature", type=float, help="Temperature for decoder.", default=0.2)
    parser.add_argument("-topp", "--topp", type=float, help="Top-p for decoder.", default=0.95)
    parser.add_argument("-ntok", "--num-tokens", type=int, help="Maximum number of tokens generated.", default=256)
    parser.add_argument("-o", "--output", required=True, type=str, help="Path to the directory to save the file.")
    parser.add_argument("-sn", "--save-name", required=True, type=str, help="Name of the file to be save.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.test_data):
        raise FileNotFoundError(f"The dataset has not been found: {args.test_data}")
    
    dataset = pd.read_pickle(args.test_data)
    dataset['input'] = dataset.apply(lambda x: x['input'][0], axis=1)
        
    llm_connector = get_llm_engine(args)
    
    results = execute_pipeline(args.pipeline, dataset, llm_connector)
    
    os.makedirs(f"{args.output}", exist_ok=True)
    df_export = pd.DataFrame.from_dict(results)
    df_export = df_export.set_index(dataset.index)
    df_export = pd.concat([df_export, dataset], axis=1)
    df_export.to_parquet(os.path.join(args.output, f"{args.save_name}.parquet.gzip"), engine="fastparquet", compression="gzip")