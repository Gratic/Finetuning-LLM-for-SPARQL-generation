from libsparqltotext import parse_script_arguments, print_header, print_additional_infos, basic_prompt
from libsparqltotext import SaveService, RegexAnswerProcessor, LLAMACPPProvider, ServerProvider, CTransformersProvider, DataProcessor, DataWorkflowController, ExportTwoFileService, DataPreparator, vLLMProvider, TransformersProvider

# Author
AUTHOR = "Alexis STRAPPAZZON"
VERSION = "0.2.3"

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
        provider = LLAMACPPProvider(args.server_address, args.server_port)
    elif args.provider == "CTRANSFORMERS":
        model_type = "llama"
        if "mistral" in args.model_path:
            model_type = "mistral"
        provider = CTransformersProvider(args.model_path, args.context_length, model_type)
    elif args.provider == "SERVER":
        provider = ServerProvider(args.server_address, args.server_port, args.completion_endpoint, args.tokenizer_endpoint)
    elif args.provider == "VLLM":
        provider = vLLMProvider(args.model_path, args.context_length)
    elif args.provider == "TRANSFORMERS":
        provider = TransformersProvider(args.model_path, args.context_length)
        
    dataPreparator = DataPreparator(provider, args.template, system_prompt, args.prompt, args.leading_answer_prompt, args.prepare_prompts)
    
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
        targets = list(dataset.loc[dataset["is_skipped"] == True].index)
        
    print_additional_infos(args, dataset, saveService, targets)
    
    regexService = RegexAnswerProcessor(args)
    dataProcessor = DataProcessor(provider=provider,
                                    answerProcessor=regexService,
                                    dataset=dataset,
                                    retry_attempts=args.retry_attempts,
                                    context_length_limit=args.context_length,
                                    prediction_size=args.prediction_size,
                                    temperature=args.temperature,
                                    print_answers=args.print_answers,
                                    print_results=args.print_results
                                    )
    generatorService = DataWorkflowController(saveService=saveService,
                                                dataProcessor=dataProcessor,
                                                dataset=dataset,
                                                generation_type=args.generation,
                                                offset=args.offset,
                                                number_of_rows=args.number_of_rows,
                                                targets=targets,
                                                verbose=args.verbose)
    exportService = ExportTwoFileService(dataset, args)
    
    
    generatorService.generate()
    exportService.export(generatorService.last_row_index)
        
    if not args.quiet:
        print("Execution successfully ended.")