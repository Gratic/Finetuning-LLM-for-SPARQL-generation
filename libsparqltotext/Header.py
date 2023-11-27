from .DataProcessor import RETRY_IF_ANSWER_CONTAINS
import argparse
import pandas as pd
from libsparqltotext import SaveService
from typing import List

# Printing constants
LINE_WIDTH = 76
LINE = "-"*LINE_WIDTH

def print_header(args: argparse.Namespace, version: str):
    if args.verbose:
        print(LINE)
        print(f"Reverse Prompt generation v{version}".center(LINE_WIDTH, ' '))
        print(LINE)
        
        print("Script Parameters")
        print("  PROVIDER".ljust(30), args.provider)
        if args.provider == "SERVER":
            print("  SERVER_ADDR".ljust(30), args.server_address)
            print("  SERVER_PORT".ljust(30), args.server_port)
            print("  SERVER_COMPLETION_ENDPOINT".ljust(30), args.endpoint)
        elif args.provider == "CTRANSFORMERS":
            print("  MODEL_PATH".ljust(30), args.model_path)
        print("  CONTEXT_LENGTH".ljust(30), args.context_length)
        print("  GENERATION_TYPE".ljust(30), args.generation)
        print("  STARTING_ROW_OFFSET".ljust(30), args.offset)
        print("  NUMBER_OF_ROWS_TO_PROCESS".ljust(30), args.number_of_rows)
        print("  MAX_NUMBER_OF_TRY_PER_PROMPT".ljust(30), args.retry_attempts)
        print("  RETRY_IF_ANSWER_CONTAINS".ljust(30), "\"" + "\", \"".join(RETRY_IF_ANSWER_CONTAINS) + "\"")
        print("  PREPARE_PROMPTS".ljust(30), args.prepare_prompts)
        print("  TARGET_ROWS".ljust(30), args.target_rows)
        print("  SYSTEM_PROMPT OR PATH".ljust(30), "PROMPT" if args.system_prompt != None or args.system_prompt != "" else "PATH")
        print("  SYSTEM_PROMPT".ljust(30), args.system_prompt)
        print("  SYSTEM_PROMPT_PATH".ljust(30), args.system_prompt_path)
        print("  OUTPUT_PATH".ljust(30), args.output_path)
        print("  QUERIES_PATH".ljust(30), args.queries_path)
        print("  NUMBER_OF_TOKEN_TO_PREDICT".ljust(30), args.prediction_size)
        print("  TEMPERATURE".ljust(30), args.temperature)
        print("  SAVE_IDENTIFIER".ljust(30), args.save_identifier)
        print("  CHECKPOINT_PATH".ljust(30), args.checkpoint_path)

def print_additional_infos(args: argparse.Namespace, df_cleaned_queries: pd.DataFrame, saveService: SaveService, targets: List[int]):
    if args.verbose:
        number_of_rows_to_process = 0
        if saveService.is_new_generation() and args.generation == "continuous":
            number_of_rows_to_process = args.number_of_rows if args.number_of_rows > 0 else len(df_cleaned_queries) - args.offset
        else:
            number_of_rows_to_process = args.number_of_rows - (saveService.last_index_row_processed - args.offset)
        
        if args.generation == "targeted" or args.generation == "skipped":
            number_of_rows_to_process = len(targets)

        print("Additional Information")
        print("  Generation state".ljust(30), "New Generation process" if saveService.is_new_generation() else "Recovered Generation process")
        print("  Number of rows to process".ljust(30), number_of_rows_to_process)
        print(LINE)