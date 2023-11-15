from tqdm import tqdm
from . import utils
from . import SaveService

RETRY_IF_ANSWER_CONTAINS = ["SELECT", "GROUP"]

class QueryGeneratorService():
    def __init__(self, provider, regexService, saveService, dataset, args) -> None:
        self.provider = provider
        self.regexService = regexService
        self.saveService = saveService
        self.dataset = dataset
        self.prompts = dataset['prompt']
        self.num_tokens = dataset['num_tokens']
        self.skipped_rows = []
        self.verbose = args.verbose
        
        self.number_of_rows = args.number_of_rows
        self.offset = args.offset
        self.last_row_index = len(self.prompts) if self.number_of_rows <= 0 else self.offset + self.number_of_rows
        self.base_retry_attempts = args.retry_attempts
        self.context_length_limit = args.context_length
        self.prediction_size = args.prediction_size
        self.temperature = args.temperature
        self.print_answers = args.print_answers
        self.print_results = args.print_results
        self.quiet = args.quiet
        
        self.starting_row = saveService.last_index_row_processed + 1 if saveService.is_resumed_generation() else self.offset
    
    def generate(self):
        if self.verbose:
            print("Generating reverse prompts... ")
            
        for row_index in tqdm(range(self.starting_row, self.last_row_index)):
            number_of_try_left = self.base_retry_attempts
            found_results = False
            too_much_tokens = False
            
            prompt = self.prompts.iat[row_index]
            num_token = self.num_tokens.iat[row_index]
            
            if num_token > self.context_length_limit:
                too_much_tokens = True
            else:
                data_json = utils.prepare_request_payload(prompt, self.prediction_size, self.temperature)
            
            while number_of_try_left != 0 and not (found_results or too_much_tokens):    
                self.provider.query(data_json)
                
                if self.print_answers:
                    print(self.provider.get_answer())
                
                results = self.regexService.extract_prompts(self.provider.get_answer())
                
                if self.print_results:
                    print(results)
                
                if not utils.are_results_acceptable(results, RETRY_IF_ANSWER_CONTAINS):
                    number_of_try_left -= 1
                    continue
                
                found_results = True
                self.dataset['result'].iat[row_index] = results
                self.dataset['full_answer'].iat[row_index] = self.provider.get_full_answer()
            
            if not found_results and not self.quiet:
                print(f"No results found for: {self.dataset.iloc[row_index].name}")
                self.skipped_rows.append(self.dataset.iloc[row_index].name)
            elif too_much_tokens and not self.quiet:
                print(f"Prompt has too much token at row: {self.dataset.iloc[row_index].name}")
                self.skipped_rows.append(self.dataset.iloc[row_index].name)
            
            self.saveService.export_save(row_index)