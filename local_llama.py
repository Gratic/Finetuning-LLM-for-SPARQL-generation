from tqdm import tqdm
import argparse
import datetime
import json
import numpy as np
import os
import pandas as pd
import re
import libsparqltotext

# Author
AUTHOR = "Alexis STRAPPAZZON"
VERSION = "0.1.6"

if __name__ == '__main__':
    args = libsparqltotext.parse_script_arguments()
    
    saveService = libsparqltotext.SaveService(args)
    saveService.load_save()
    
    provider = None
    if args.provider == "SERVER":
        provider = libsparqltotext.ServerProvider()
    elif args.provider == "CTRANSFORMERS":
        provider = libsparqltotext.CTransformersProvider()
    
    libsparqltotext.print_header(args, VERSION)
    
    dataset = None
    if saveService.is_new_generation():    
        dataset = libsparqltotext.load_and_prepare_queries(libsparqltotext.basic_prompt, args.queries_path, args.system_prompt, args.prepare_prompts)
    else:
        dataset = saveService.dataset
        
    libsparqltotext.print_additional_infos(args, dataset)
    
    regexService = libsparqltotext.RegexService(args)
    
    generatorService = libsparqltotext.QueryGeneratorService(provider, regexService, saveService, dataset, args)
    
    exportService = libsparqltotext.ExportService(dataset, generatorService.skipped_rows, args)
    exportService.export(generatorService.last_row_index)
        
    if not args.quiet:
        print("Execution successfully ended.")