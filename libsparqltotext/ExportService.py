import json
import os
import datetime
from abc import ABC, abstractmethod
import pandas as pd
import argparse
from typing import Any

class BaseExportService(ABC):
    def __init__(self, dataset: pd.DataFrame, args: argparse.Namespace):
        self.dataset = dataset
        self.output_path = args.output_path
        self.args = args
        
        os.makedirs(self.output_path, exist_ok=True)
    
    @abstractmethod
    def export(self, last_row_number: int):
        pass
    
    def _make_summary(self) -> dict[str, Any]:
        return {
            "PROVIDER": self.args.provider,
            "SERVER_ADDR": self.args.server_address,
            "SERVER_PORT": self.args.server_port,
            "SERVER_COMPLETION_ENDPOINT": self.args.completion_endpoint,
            "SERVER_TOKENIZER_ENDPOINT": self.args.tokenizer_endpoint,
            "MODEL_PATH": self.args.model_path,
            "CONTEXT_LENGTH": self.args.context_length,
            "GENERATION_TYPE": self.args.generation,
            "STARTING_ROW_OFFSET": self.args.offset,
            "NUMBER_OF_ROWS_TO_PROCESS": self.args.number_of_rows,
            "MAX_NUMBER_OF_TRY_PER_PROMPT": self.args.retry_attempts,
            "PREPARE_PROMPTS": self.args.prepare_prompts,
            "TARGET_ROWS": self.args.target_rows,
            # "RETRY_IF_ANSWER_CONTAINS": self.args.retry_if_answer_contains,
            "SYSTEM_PROMPT": self.args.system_prompt,
            "SYSTEM_PROMPT_PATH": self.args.system_prompt_path,
            "OUTPUT_PATH": self.args.output_path,
            "DATASET_PATH": self.args.queries_path,
            "NUMBER_OF_TOKEN_TO_PREDICT": self.args.prediction_size,
            "TEMPERATURE": self.args.temperature,
            "QUIET": self.args.quiet,
            "VERBOSE": self.args.verbose,
            "PRINT_ANSWERS": self.args.print_answers,
            "PRINT_RESULTS": self.args.print_results,
            "SAVE_IDENTIFIER": self.args.save_identifier,
            "SAVE_PATH": self.args.checkpoint_path,
            "NUMBER_OF_SKIPPED_ROWS": len(self.dataset.loc[self.dataset["is_skipped"] == True])
    }

class ExportOneFileService(BaseExportService):
    def __init__(self, dataset, args) -> None:
        super().__init__(dataset, args)
    
    def export(self, last_row_number):
        if self.args.verbose:
            print("Printing dataset... ", end="")
        
        dataframe_json_dump = self.dataset.iloc[self.args.offset:last_row_number].to_json()
        summary_json_dump = json.dumps(self._make_summary())
        
        export_dict = dict()
        export_dict['dataset'] = dataframe_json_dump
        export_dict['summary'] = summary_json_dump
        
        export_json = json.dumps(export_dict)
        
        filename = self.output_path + datetime.datetime.now().strftime('%Y%m%d-%H%M')
        
        with open(f"{filename}_results.json", 'w') as f:
            f.write(export_json)
            
        if self.args.verbose:
            print("Done.")

class ExportTwoFileService(BaseExportService):
    def __init__(self, dataset, args) -> None:
        super().__init__(dataset, args)
    
    def export(self, last_row_number):
        if self.args.verbose:
            print("Printing dataset... ", end="")
        
        dataframe_json_dump = self.dataset.iloc[self.args.offset:last_row_number].to_json()
        summary_json_dump = json.dumps(self._make_summary())
        
        filename = self.output_path + datetime.datetime.now().strftime('%Y%m%d-%H%M')
        
        with open(f"{filename}_dataset.json", 'w') as f:
            f.write(dataframe_json_dump)
            
        with open(f"{filename}_summary.json", 'w') as f:
            f.write(summary_json_dump)
            
        if self.args.verbose:
            print("Done.")
    
    