import os
import json

class SaveService():
    def __init__(self, args) -> None:
        self.id = args.save_identifier
        self.checkpoint_path = args.checkpoint_path
        self.filepath = self.checkpoint_path + f"{self.id}.chk"
        
        self.args = args
        self.dataset = None
        self.last_index_row_processed = 0
        
        self.is_resumed = False
        
        os.makedirs(self.checkpoint_path, exist_ok=True)
    
    def load_save(self):
        if os.path.exists(self.filepath):
            save_checkpoint_data = None
            with open(self.checkpoint_path + f"{self.id}.chk", 'r') as f:
                save_checkpoint_data = json.load(f)
            
            self.args = save_checkpoint_data['args']
            self.dataset = save_checkpoint_data['dataset']
            self.last_index_row_processed = save_checkpoint_data['last_index_row_processed']
            self.is_resumed = True
        return (self.args, self.dataset, self.last_index_row_processed)
    
    def export_save(self, last_index_row_processed):
        checkpoint_dict = dict()
        checkpoint_dict['args'] = self.args
        checkpoint_dict['dataset'] = self.dataset
        checkpoint_dict['last_index_row_processed'] = last_index_row_processed
        
        checkpoint_json = json.dumps(checkpoint_dict)
        
        with open(self.checkpoint_path + f"{self.id}.chk", 'w') as f:
            f.write(checkpoint_json)
            
    def is_resumed_generation(self):
        return self.is_resumed
    
    def is_new_generation(self):
        return not self.is_resumed