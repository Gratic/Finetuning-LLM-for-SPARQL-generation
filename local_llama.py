from libsparqltotext import parse_script_arguments, print_header, print_additional_infos, basic_prompt, load_and_prepare_queries
from libsparqltotext import SaveService, RegexAnswerProcessor, LLAMACPPProvider, CTransformersProvider, DataProcessor, DataWorkflowController, ExportThreeFileService, DataPreparator

# Author
AUTHOR = "Alexis STRAPPAZZON"
VERSION = "0.2.0"

if __name__ == '__main__':
    args = parse_script_arguments()
    print_header(args, VERSION)
    
    saveService = SaveService(args)
    regexService = RegexAnswerProcessor(args)
    
    system_prompt = args.system_prompt
    if args.system_prompt == None or args.system_prompt == "":
        with open(args.system_prompt_path, "r") as f:
            system_prompt = f.read()
    
    provider = None
    if args.provider == "SERVER":
        provider = LLAMACPPProvider(args.server_address, args.server_port)
    elif args.provider == "CTRANSFORMERS":
        provider = CTransformersProvider(args.model_path, args.context_length)
        
    dataPreparator = DataPreparator(provider, basic_prompt, system_prompt, args.prepare_prompts)
    
    dataset = None
    saveService.load_save()
    if saveService.is_new_generation():    
        dataPreparator.load_dataframe(args.queries_path)
        dataset = dataPreparator.get_dataset()
        saveService.dataset = dataset
    else:
        dataset = saveService.dataset
    
    targets = [int(x) for x in args.target_rows.split(",")]
    
    dataProcessor = DataProcessor(provider=provider,
                                    answerProcessor=regexService,
                                    dataset=dataset,
                                    retry_attempts=args.retry_attempts,
                                    context_length_limit=args.context_length,
                                    prediction_size=args.prepare_prompts,
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
                                                verbose=args.verbose,
                                                quiet=args.quiet)
    exportService = ExportThreeFileService(dataset, args)
    
    print_additional_infos(args, dataset, saveService)
    
    generatorService.generate()
    exportService.export(generatorService.last_row_index)
        
    if not args.quiet:
        print("Execution successfully ended.")