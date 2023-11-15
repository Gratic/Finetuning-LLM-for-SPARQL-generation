# Printing constants
LINE_WIDTH = 76
LINE = "-"*LINE_WIDTH

def print_header(args, version):
    if args.verbose:
        print(LINE)
        print(f"Reverse Prompt generation v{version}".center(LINE_WIDTH, ' '))
        print(LINE)
        
        print("Script Parameters")
        print("  PROVIDER".ljust(30), args.provider)
        if args.provider == "SERVER":
            print("  SERVER_ADDR".ljust(30), args.server_addr)
            print("  SERVER_PORT".ljust(30), args.server_port)
            print("  SERVER_COMPLETION_ENDPOINT".ljust(30), args.server_completion_endpoint)
        elif args.provider == "CTRANSFORMERS":
            print("  MODEL_PATH".ljust(30), args.ct_model_path)
        print("  STARTING_ROW_OFFSET".ljust(30), args.starting_row_offset)
        print("  NUMBER_OF_ROWS_TO_PROCESS".ljust(30), args.number_of_rows_to_process)
        print("  SYSTEM_PROMPT".ljust(30), args.system_prompt)
        print("  MAX_NUMBER_OF_TRY_PER_PROMPT".ljust(30), args.max_number_of_try_per_prompt)
        print("  RETRY_IF_ANSWER_CONTAINS".ljust(30), "\"" + "\", \"".join(args.retry_if_answer_contains) + "\"")
        print("  PREPARE_PROMPTS".ljust(30), args.prepare_prompts)
        print("  OUTPUT_PATH".ljust(30), args.output_path)
        print("  CLEANED_QUERIES_PATH".ljust(30), args.queries_path)
        print("  NUMBER_OF_TOKEN_TO_PREDICT".ljust(30), args.number_of_token_to_predict)
        print("  TEMPERATURE".ljust(30), args.temperature)

def print_additional_infos(args, df_cleaned_queries):
    if args.verbose:
        print("Additional Information")
        print("  Number of rows to process".ljust(30), args.number_of_rows_to_process if args.number_of_rows_to_process > 0 else len(df_cleaned_queries) - args.starting_row_offset)
        print(LINE)