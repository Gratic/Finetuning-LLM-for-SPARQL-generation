import argparse

# Connections options
PROVIDER = "LLAMACPP"
SERVER_ADDR = "127.0.0.1"
SERVER_PORT = "8080"
SERVER_COMPLETION_ENDPOINT = "/completion"
SERVER_TOKENIZER_ENDPOINT = "/tokenize"
CT_MODEL_PATH = ""
CT_CONTEXT_LENGTH = 4096
 
# Dataset processing options
GENERATION_TYPE = "continuous"
STARTING_ROW_OFFSET = 0
NUMBER_OF_ROWS_TO_PROCESS = 0 # Rows [STARTING_ROW_OFFSET:STARTING_ROW_OFFSET+NUMBER_OF_ROWS_TO_PROCESS-1] will be processed. <0 will do [STARTING_ROW_OFFSET:len(data)-1].
TEMPLATE = "[INST] [system_prompt] [data] [prompt] [/INST] [lead_answer_prompt]"
SYSTEM_PROMPT = "<<SYS>>This is a conversation between User and Llama, a friendly chatbot. Llama is helpful, kind, honest, good at writing, and never fails to answer any requests immediately and with precision.<</SYS>>\n"
PROMPT = "User: Read QUERY, DESCRIPTION and CONTEXT. There is a machine capable of writing the given QUERY if we ask it the right prompt. Please do not include parts of QUERY in your answers. Give a list of 3 prompts that would give QUERY."
LEAD_ANS_PROMPT = "Llama: "
MAX_NUMBER_OF_TRY_PER_PROMPT = 25
PREPARE_PROMPTS = "auto"
TARGET_ROWS = ""

# Dataset output options
OUTPUT_PATH = "./outputs/generated_prompts/"
QUERIES_PATH = "./datasets/final_queries_v1.1.json"

# Prompt processing/Generation options
NUMBER_OF_TOKEN_TO_PREDICT = 256
TEMPERATURE = 0.4 # (default = 0.8)

SAVE_ID = "0"
SAVE_CHECKPOINT_PATH = "./outputs/checkpoints/"
ARGUMENT = "cli"

def parse_script_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pv", "--provider", type=str, help=f"Who completes the answer (default={PROVIDER}).", choices=["SERVER", "CTRANSFORMERS", "LLAMACPP", "VLLM"], default=PROVIDER)
    parser.add_argument("-saddr", "--server-address", type=str, help=f"IP address or URL of the server that has the LLM API endpoint if the provider is SERVER (default={SERVER_ADDR}).", default=SERVER_ADDR)
    parser.add_argument("-sport", "--server-port", type=str, help=f"Port to ask for connection with the server_address if the provider is SERVER (default={SERVER_PORT}).", default=SERVER_PORT)
    parser.add_argument("-ec", "--completion-endpoint", type=str, help=f"Endpoint of the completion API if the provider is SERVER (default={SERVER_COMPLETION_ENDPOINT}).", default=SERVER_COMPLETION_ENDPOINT)
    parser.add_argument("-et", "--tokenizer-endpoint", type=str, help=f"Endpoint of the tokenizer API if the provider is SERVER (default={SERVER_TOKENIZER_ENDPOINT}).", default=SERVER_TOKENIZER_ENDPOINT)
    parser.add_argument("-mp", "--model-path", type=str, help=f"Path to the model if the provider is CTRANSFORMERS (default={CT_MODEL_PATH}).", default=CT_MODEL_PATH)
    parser.add_argument("-nctx", "--context-length", type=int, help=f"Size of the Context length of the model (default={CT_CONTEXT_LENGTH}).", default=CT_CONTEXT_LENGTH)
    
    parser.add_argument("-g", "--generation", type=str, help=f"Generation type: \"continuous\" will start at \"offset\" and go until the end or \"number-of-rows\", \"targeted\" will only do the rows number specified by \"target-rows\", \"skipped\" will recover the skipped rows according to \"save-identifier\" id and process them again. (default={GENERATION_TYPE}).", choices=["continuous", "targeted", "skipped"], default=GENERATION_TYPE)
    parser.add_argument("-o", "--offset", type=int, help=f"Offset the starting row processed by the script. (default={STARTING_ROW_OFFSET}).", default=STARTING_ROW_OFFSET)
    parser.add_argument("-n", "--number-of-rows", type=int, help=f"Rows [offset:offset+n-1] will be processed. <0 will do [offset:number of rows in the data-1] (default={NUMBER_OF_ROWS_TO_PROCESS}), will process all rows).", default=NUMBER_OF_ROWS_TO_PROCESS)
    parser.add_argument("-r", "--retry-attempts", type=int, help=f"Number of retries to attempt on generating prompts from one row query. A failure happen when the result from the LLM does not satisfy the constraints. 0 means no retry, -1 means retry until constraint satisfaction, n positive tells the number of attempts (default={MAX_NUMBER_OF_TRY_PER_PROMPT}).", default=MAX_NUMBER_OF_TRY_PER_PROMPT)
    parser.add_argument("-pp", "--prepare-prompts", type=str, help=f"Should the script prepare prompts? \"auto\" will detect if there is a \"prompt\" column in the dataset. \"no\" will not do anything, and \"yes\" will create a prompt column and make them (default={PREPARE_PROMPTS}).", choices=["auto", "yes", "no"], default=PREPARE_PROMPTS)
    parser.add_argument("-tr", "--target-rows", type=str, help=f"Comma separated string of row number to be processed. Used only when \"generation\"=\"targeted\" (default={TARGET_ROWS}).", choices=["auto", "yes", "no"], default=TARGET_ROWS)
    
    parser.add_argument("-tp", "--template", type=str, help=f"The template to use, a default template is automatically given (default={TEMPLATE}).", default=TEMPLATE)
    parser_system_prompt = parser.add_mutually_exclusive_group()
    parser_system_prompt.add_argument("-sys", "--system-prompt", type=str, help=f"The system prompt to use, a default system prompt is automatically given (default={SYSTEM_PROMPT}).", default=SYSTEM_PROMPT)
    parser_system_prompt.add_argument("-sysp", "--system-prompt-path", type=str, help=f"Path to the system prompt file which should be a normal text file, a default system prompt is automatically given (default=\"\").", default="")
    parser.add_argument("-pt", "--prompt", type=str, help=f"The prompt to use, a default system prompt is automatically given (default={PROMPT}).", default=PROMPT)
    parser.add_argument("-la", "--leading-answer-prompt", type=str, help=f"The leading answer prompt to use, a default system prompt is automatically given (default={LEAD_ANS_PROMPT}).", default=LEAD_ANS_PROMPT)
    
    parser.add_argument("-out", "--output-path", type=str, help=f"Path to the directory where to output file (default={OUTPUT_PATH}).", default=OUTPUT_PATH)
    parser.add_argument("-p", "--queries-path", type=str, help=f"Path to the queries' file (default={QUERIES_PATH}).", default=QUERIES_PATH)
    
    parser.add_argument("-np", "--prediction-size", type=int, help=f"Define the number of token maximum generated by the LLM. The LLM can try to match this given size. Higher number might give more accurate result (default={NUMBER_OF_TOKEN_TO_PREDICT}).", default=NUMBER_OF_TOKEN_TO_PREDICT)
    parser.add_argument("-t", "--temperature", type=float, help=f"Temperature is a parameter of randomness of the output generated by the LLM, check google for more information (default={TEMPERATURE}).", default=TEMPERATURE)
    
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument("-q", "--quiet", action="store_true" ,help=f"Disable any output from the script.")
    parser_verbosity.add_argument("-v", "--verbose", action="store_true" ,help=f"Show more information about the process.")
    
    parser.add_argument("-pa", "--print-answers", action="store_true" ,help=f"Print answers from the LLM.")
    parser.add_argument("-pr", "--print-results", action="store_true" ,help=f"Print results extracted from the answer of LLM.")
    
    parser.add_argument("-id", "--save-identifier", type=str, help=f"Save ID, used to make checkpoint and resume at checkpoints (default={SAVE_ID}).", default=SAVE_ID)
    parser.add_argument("-cp", "--checkpoint-path", type=str, help=f"Path to save checkpoint (default={SAVE_CHECKPOINT_PATH}).", default=SAVE_CHECKPOINT_PATH)
    parser.add_argument("-a", "--argument", type=str, help=f"By default will give priority of the arguments passed down by the cli rather than those stored in the checkpoint file (default={ARGUMENT}).", choices=["cli", "checkpoint"], default=ARGUMENT)
    
    return parser.parse_args()