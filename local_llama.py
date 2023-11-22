from libsparqltotext import parse_script_arguments, print_header, print_additional_infos, basic_prompt, load_and_prepare_queries
from libsparqltotext import SaveService, RegexAnswerProcessor, ServerProvider, CTransformersProvider, DataProcessor, DataWorkflowController, ExportThreeFileService

# Author
AUTHOR = "Alexis STRAPPAZZON"
VERSION = "0.1.8"

if __name__ == '__main__':
    args = parse_script_arguments()
    print_header(args, VERSION)
    
    saveService = SaveService(args)
    regexService = RegexAnswerProcessor(args)
    
    provider = None
    if args.provider == "SERVER":
        provider = ServerProvider(args)
    elif args.provider == "CTRANSFORMERS":
        provider = CTransformersProvider(args)
    
    saveService.load_save()
    dataset = None
    if saveService.is_new_generation():    
        dataset = load_and_prepare_queries(basic_prompt, args.queries_path, args.system_prompt, args.prepare_prompts)
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
    generatorService = DataWorkflowController(provider=provider,
                                                             saveService=saveService,
                                                             dataProcessor=dataProcessor,
                                                             dataset=dataset,
                                                             generation_type=args.generation,
                                                             offset=args.offset,
                                                             number_of_rows=args.number_of_rows,
                                                             targets=targets,
                                                             verbose=args.verbose,
                                                             quiet=args.quiet)
    exportService = ExportThreeFileService(dataset, generatorService.skipped_rows, args)
    
    print_additional_infos(args, dataset, saveService)
    
    generatorService.generate()
    exportService.export(generatorService.last_row_index)
        
    if not args.quiet:
        print("Execution successfully ended.")