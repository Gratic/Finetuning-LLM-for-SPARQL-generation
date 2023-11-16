import libsparqltotext

# Author
AUTHOR = "Alexis STRAPPAZZON"
VERSION = "0.1.7"

if __name__ == '__main__':
    args = libsparqltotext.parse_script_arguments()
    libsparqltotext.print_header(args, VERSION)
    
    saveService = libsparqltotext.SaveService(args)
    regexService = libsparqltotext.RegexService(args)
    
    provider = None
    if args.provider == "SERVER":
        provider = libsparqltotext.ServerProvider(args)
    elif args.provider == "CTRANSFORMERS":
        provider = libsparqltotext.CTransformersProvider(args)
    
    saveService.load_save()
    dataset = None
    if saveService.is_new_generation():    
        dataset = libsparqltotext.load_and_prepare_queries(libsparqltotext.basic_prompt, args.queries_path, args.system_prompt, args.prepare_prompts)
        saveService.dataset = dataset
    else:
        dataset = saveService.dataset
        
    generatorService = libsparqltotext.QueryGeneratorService(provider, regexService, saveService, dataset, args)
    exportService = libsparqltotext.ExportThreeFileService(dataset, generatorService.skipped_rows, args)
    
    libsparqltotext.print_additional_infos(args, dataset, saveService)
    
    generatorService.generate()
    exportService.export(generatorService.last_row_index)
        
    if not args.quiet:
        print("Execution successfully ended.")