import os
import json
import pandas as pd
import argparse


class SaveService():
    def __init__(self, args: argparse.Namespace) -> None:
        self.id: str = args.save_identifier
        self.checkpoint_path: str = args.checkpoint_path
        self.filepath: str = self.checkpoint_path + f"{self.id}.chk"
        
        self.args: argparse.Namespace = args
        self.dataset: pd.DataFrame = None
        self.last_index_row_processed: int = 0
        
        self.is_resumed = False
        
        os.makedirs(self.checkpoint_path, exist_ok=True)
    
    def load_save(self) -> tuple[argparse.Namespace, pd.DataFrame, int]:
        if os.path.exists(self.filepath):
            save_checkpoint_data = None
            with open(self.checkpoint_path + f"{self.id}.chk", 'r') as f:
                save_checkpoint_data = json.load(f)
            
            self.args.__dict__ = save_checkpoint_data['args']
            self.dataset = pd.read_json(save_checkpoint_data['dataset'])
            self.last_index_row_processed = save_checkpoint_data['last_index_row_processed']
            self.is_resumed = True
        return (self.args, self.dataset, self.last_index_row_processed)
    
    def export_save(self, last_index_row_processed) -> None:
        checkpoint_dict = dict()
        checkpoint_dict['args'] = self.args.__dict__
        checkpoint_dict['dataset'] = self.dataset.to_json()
        checkpoint_dict['last_index_row_processed'] = last_index_row_processed
        
        checkpoint_json = json.dumps(checkpoint_dict)
        
        with open(self.checkpoint_path + f"{self.id}.chk", 'w') as f:
            f.write(checkpoint_json)
            
    def is_resumed_generation(self) -> bool:
        return self.is_resumed
    
    def is_new_generation(self) -> bool:
        return not self.is_resumed