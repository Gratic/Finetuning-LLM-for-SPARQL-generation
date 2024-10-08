import sys
from pathlib import Path
sys.path.append(Path("modules").absolute().__str__())

from libsparqltotext import (
    CTransformersProvider,
    DataPreparator,
    DataProcessor,
    DataWorkflowController,
    ExportTwoFileService,
    LLAMACPPProvider,
    parse_script_arguments,
    print_additional_infos,
    print_header,
    RegexAnswerProcessor,
    SaveService,
    ServerProvider,
    TransformersProvider,
    TransformersProviderv2,
    vLLMProvider,
    OpenAIProvider,
)

# Author
AUTHOR = "Alexis STRAPPAZZON"
VERSION = "0.2.6"

if __name__ == '__main__':
    args = parse_script_arguments()
    print_header(args, VERSION)
    
    saveService = SaveService(args)
    
    system_prompt = args.system_prompt
    if args.system_prompt == None or args.system_prompt == "":
        with open(args.system_prompt_path, "r") as f:
            system_prompt = f.read()
    
    provider = None
    if args.provider == "LLAMACPP":
        provider = LLAMACPPProvider(args.server_address, args.server_port, temperature=args.temperature, n_predict=args.prediction_size)
    elif args.provider == "CTRANSFORMERS":
        model_type = "llama"
        if "mistral" in args.model_path:
            model_type = "mistral"
        provider = CTransformersProvider(args.model_path, args.context_length, model_type, temperature=args.temperature, n_predict=args.prediction_size)
    elif args.provider == "SERVER":
        provider = ServerProvider(args.server_address, args.server_port, args.completion_endpoint, args.tokenizer_endpoint, temperature=args.temperature, n_predict=args.prediction_size)
    elif args.provider == "VLLM":
        provider = vLLMProvider(args.model_path, args.context_length, temperature=args.temperature, n_predict=args.prediction_size)
    # TODO: add topp as parser argument
    elif args.provider == "TRANSFORMERS":
        provider = TransformersProvider(args.model_path, args.context_length, top_p=0.8, temperature=args.temperature, n_predict=args.prediction_size)
    elif args.provider == "TRANSFORMERSv2":
        provider = TransformersProviderv2(args.model_path, args.context_length, top_p=0.98, temperature=args.temperature, n_predict=args.prediction_size)
    elif args.provider == "OpenAI":
        provider = OpenAIProvider(args.model_path, api_key=args.api_key, top_p=0.95, temperature=args.temperature, n_predict=args.prediction_size)
        
    dataPreparator = DataPreparator(
        provider=provider,
        template=args.template,
        system_prompt=system_prompt,
        prompt=args.prompt,
        lead_answer_prompt=args.leading_answer_prompt,
        prompt_preparation=args.prepare_prompts,
        prefix=args.prefix,
        query_column=args.query_column)
    
    dataset = None
    saveService.load_save()
    if saveService.is_new_generation():    
        dataPreparator.load_dataframe(args.queries_path)
        dataset = dataPreparator.get_dataset()
        saveService.dataset = dataset
    else:
        dataset = saveService.dataset
        if args.argument == "checkpoint":
            args = saveService.args
    
    targets = None
    if args.generation == "targeted":
        targets = [int(x) for x in args.target_rows.split(",")]
    elif args.generation == "skipped":
        targets = list(dataset.loc[dataset[f"{args.prefix}is_skipped"] == True].index)
        
    print_additional_infos(args, dataset, saveService, targets)
    
    regexService = RegexAnswerProcessor(args)
    dataProcessor = DataProcessor(provider=provider,
                                    answerProcessor=regexService,
                                    dataset=dataset,
                                    retry_attempts=args.retry_attempts,
                                    context_length_limit=args.context_length,
                                    print_answers=args.print_answers,
                                    print_results=args.print_results,
                                    query_column=args.query_column,
                                    prefix=args.prefix
                                    )
    generatorService = DataWorkflowController(saveService=saveService,
                                                dataProcessor=dataProcessor,
                                                dataset=dataset,
                                                generation_type=args.generation,
                                                offset=args.offset,
                                                number_of_rows=args.number_of_rows,
                                                targets=targets,
                                                verbose=args.verbose,
                                                prefix=args.prefix)
    exportService = ExportTwoFileService(dataset, args)
    
    
    generatorService.generate()
    exportService.export(generatorService.last_row_index)
        
    if not args.quiet:
        print("Execution successfully ended.")