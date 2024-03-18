import unittest
from scripts.sft_peft import compute_metrics, tokenizer
import json
from pathlib import Path
import numpy as np
from typing import Tuple
import scripts.sft_peft
from transformers import AutoTokenizer
import evaluate
import nltk


def load_data(name: str):
    if name == "no_acceptable_queries":
        path = Path("tests/scriptstests/data/sft_peft_compute_metrics_no_acceptable_queries.json")
    elif name == "execute_ok":
        path = Path("tests/scriptstests/data/sft_peft_compute_metrics_execute_ok.json")
    data = json.loads(path.read_text())
    return data

def data_to_compute_metrics_args(data: dict) -> Tuple[np.ndarray, np.ndarray]:
    return (np.array(data['preds']), np.array(data['labels']))

class MockTokenizer():
    def __init__(self, pad_token_id: int = 0) -> None:
        self.pad_token_id = pad_token_id

class SFTPeftTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
    
    def test_load_data_no_acceptable_queries(self):
        data = load_data("no_acceptable_queries")
        
        self.assertIsInstance(data, dict)
        self.assertListEqual(['preds', 'labels', 'decoded_preds', 'decoded_labels'], list(data.keys()))
        self.assertEqual(5, len(data['preds']))
        self.assertEqual(5, len(data['labels']))
    
    def test_load_data_execute_ok(self):
        data = load_data("execute_ok")
        
        self.assertIsInstance(data, dict)
        self.assertListEqual(['preds', 'labels'], list(data.keys()))
        self.assertEqual(5, len(data['preds']))
        self.assertEqual(5, len(data['labels']))
    
    def test_compute_metrics_with_no_acceptable_queries(self):
        data = load_data("no_acceptable_queries")
        
        args = data_to_compute_metrics_args(data)
        
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        tokenizer.pad_token = tokenizer.unk_token
        nltk.download("punkt", quiet=True)
        scripts.sft_peft.tokenizer = tokenizer
        scripts.sft_peft.rouge_metric = evaluate.load("rouge")
        scripts.sft_peft.bleu_metric = evaluate.load("bleu")
        scripts.sft_peft.meteor_metric = evaluate.load("meteor")
        scripts.sft_peft.start_tag = "`sparql\n"
        scripts.sft_peft.end_tag = "`"
        
        compute_metrics(args)
        
    def test_compute_metrics_with_execute_ok(self):
        data = load_data("execute_ok")
        
        args = data_to_compute_metrics_args(data)
        
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        tokenizer.pad_token = tokenizer.unk_token
        nltk.download("punkt", quiet=True)
        scripts.sft_peft.tokenizer = tokenizer
        scripts.sft_peft.rouge_metric = evaluate.load("rouge")
        scripts.sft_peft.bleu_metric = evaluate.load("bleu")
        scripts.sft_peft.meteor_metric = evaluate.load("meteor")
        scripts.sft_peft.start_tag = "`sparql\n"
        scripts.sft_peft.end_tag = "`"
        
        print(compute_metrics(args))