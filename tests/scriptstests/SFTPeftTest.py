import sys
from pathlib import Path
sys.path.append(Path("modules").absolute().__str__())

import unittest
from scripts.sft_peft import (
    compute_metrics,
    is_query_format_acceptable,
    generate_instruction_prompt,
    parse_args,
    extract_query,
)
import json
from pathlib import Path
import numpy as np
from typing import Tuple
import scripts.sft_peft
from transformers import AutoTokenizer
import evaluate
import nltk
from prompts_template import BASE_MISTRAL_TEMPLATE, BASE_BASIC_INSTRUCTION
from libwikidatallm.TemplateLLMQuerySender import TemplateLLMQuerySender
from parameterized import parameterized
from itertools import product
import argparse

def _get_name_for_get_args(testcase_func, param_num, param):
    return f"{testcase_func.__name__}_{param_num}"

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
        
        the_args = argparse.Namespace()
        the_args.__setattr__("output", "tests/scriptstests/data")
        the_args.__setattr__("save_name", "no_acceptable")
        scripts.sft_peft.args = the_args
        
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
        
        compute_metrics(args)
    
    def test_is_query_format_acceptable_with_good_query(self):
        scripts.sft_peft.start_tag = "[query]"
        scripts.sft_peft.end_tag = "[/query]"
        
        query = """[query]SELECT start of the query[/query]"""
        
        self.assertTrue(is_query_format_acceptable(query))
        
    def test_is_query_format_acceptable_with_bad_starting_token(self):
        scripts.sft_peft.start_tag = "[query]"
        scripts.sft_peft.end_tag = "[/query]"
        
        query = """[not the starting token]start of the query[/query]"""
        
        self.assertFalse(is_query_format_acceptable(query))
        
    def test_is_query_format_acceptable_with_bad_ending_token(self):
        scripts.sft_peft.start_tag = "[query]"
        scripts.sft_peft.end_tag = "[/query]"
        
        query = """[query]start of the query[not the ending token]"""
        
        self.assertFalse(is_query_format_acceptable(query))
        
    def test_is_query_format_acceptable_with_bad_both_tokens(self):
        scripts.sft_peft.start_tag = "[query]"
        scripts.sft_peft.end_tag = "[/query]"
        
        query = """[not the starting token]start of the query[not the ending token]"""
        
        self.assertFalse(is_query_format_acceptable(query))
        
    def test_is_query_format_acceptable_with_no_token(self):
        scripts.sft_peft.start_tag = "[query]"
        scripts.sft_peft.end_tag = "[/query]"
        
        query = """start of the query"""
        
        self.assertFalse(is_query_format_acceptable(query))
        
    def test_is_query_format_acceptable_with_empty_query(self):
        scripts.sft_peft.start_tag = "[query]"
        scripts.sft_peft.end_tag = "[/query]"
        
        query = """"""
        
        self.assertFalse(is_query_format_acceptable(query))
        
    def test_is_query_format_acceptable_with_tokens_but_empty_query(self):
        scripts.sft_peft.start_tag = "[query]"
        scripts.sft_peft.end_tag = "[/query]"
        
        query = """[query][/query]"""
        
        self.assertFalse(is_query_format_acceptable(query))
        
    def test_is_query_format_acceptable_no_token_but_empty_query(self):
        scripts.sft_peft.start_tag = ""
        scripts.sft_peft.end_tag = ""
        
        query = """"""
        
        self.assertFalse(is_query_format_acceptable(query))
        
    def test_is_query_format_acceptable_no_token_but_not_query(self):
        scripts.sft_peft.start_tag = ""
        scripts.sft_peft.end_tag = ""
        
        query = """not query"""
        
        self.assertFalse(is_query_format_acceptable(query))
        
    def test_is_query_format_acceptable_no_token_but_query(self):
        scripts.sft_peft.start_tag = ""
        scripts.sft_peft.end_tag = ""
        
        query = """SELECT not query"""
        
        self.assertTrue(is_query_format_acceptable(query))
        
    def test_is_query_format_acceptable_no_token_but_query_select(self):
        scripts.sft_peft.start_tag = ""
        scripts.sft_peft.end_tag = ""
        
        query = """select not query"""
        
        self.assertTrue(is_query_format_acceptable(query))
        
    def test_is_query_format_acceptable_no_token_but_query_prefix(self):
        scripts.sft_peft.start_tag = ""
        scripts.sft_peft.end_tag = ""
        
        query = """prefix not query"""
        
        self.assertTrue(is_query_format_acceptable(query))
        
    def test_is_query_format_acceptable_no_token_but_query_PREFIX(self):
        scripts.sft_peft.start_tag = ""
        scripts.sft_peft.end_tag = ""
        
        query = """PREFIX not query"""
        
        self.assertTrue(is_query_format_acceptable(query))
        
    def test_generate_instruction_prompt_normal_with_query_token(self):
        template = BASE_MISTRAL_TEMPLATE
        scripts.sft_peft.templater = TemplateLLMQuerySender(None, template, start_seq='[', end_seq=']')
        
        scripts.sft_peft.start_tag = "[query]"
        scripts.sft_peft.end_tag = "[/query]"
        
        prompt = "the prompt"
        target = "the target"
        
        expected = f"""[INST] {BASE_BASIC_INSTRUCTION} the prompt [/INST] [query]the target[/query]"""
        
        self.assertEqual(expected, generate_instruction_prompt(prompt, target))
        
    def test_generate_instruction_prompt_normal_with_sparql_token(self):
        template = BASE_MISTRAL_TEMPLATE
        scripts.sft_peft.templater = TemplateLLMQuerySender(None, template, start_seq='[', end_seq=']')
        
        scripts.sft_peft.start_tag = "`sparql\n"
        scripts.sft_peft.end_tag = "`"
        
        prompt = "the prompt"
        target = "the target"
        
        expected = f"""[INST] {BASE_BASIC_INSTRUCTION} the prompt [/INST] `sparql
the target`"""
        
        self.assertEqual(expected, generate_instruction_prompt(prompt, target))
    
    def test_generate_instruction_prompt_normal_with_no_token(self):
        template = BASE_MISTRAL_TEMPLATE
        scripts.sft_peft.templater = TemplateLLMQuerySender(None, template, start_seq='[', end_seq=']')
        
        scripts.sft_peft.start_tag = ""
        scripts.sft_peft.end_tag = ""
        
        prompt = "the prompt"
        target = "the target"
        
        expected = f"""[INST] {BASE_BASIC_INSTRUCTION} the prompt [/INST] the target"""
        
        self.assertEqual(expected, generate_instruction_prompt(prompt, target))
    
    @parameterized.expand([
        ("[query]", "[/query]"),
        ("[sparql]", "[/sparql]"),
        ("<query>", "</query>"),
        ("<sparql>", "</sparql>"),
        ("`sparql\n", "`"),
        ("`\n", "`"),
        ("", ""),
    ],
    name_func=_get_name_for_get_args)
    def test_generate_instruction_prompt(self, start_tag, end_tag):
        template = BASE_MISTRAL_TEMPLATE
        scripts.sft_peft.templater = TemplateLLMQuerySender(None, template, start_seq='[', end_seq=']')
        
        scripts.sft_peft.start_tag = start_tag
        scripts.sft_peft.end_tag = end_tag
        
        prompt = "the prompt"
        target = "the target"
        
        expected = f"""[INST] {BASE_BASIC_INSTRUCTION} the prompt [/INST] {start_tag}the target{end_tag}"""
        
        self.assertEqual(expected, generate_instruction_prompt(prompt, target))
    
    def test_parse_args_testing_start_end_tokens_default(self):
        list_args = [
            "--train-data", "dont care"
            ]
        
        args = parse_args(list_args)
        
        self.assertEqual(args.start_tag, "[query]")
        self.assertEqual(args.end_tag, "[/query]")
    
    @parameterized.expand([
        ("[query]", "[/query]", "[query]", "[/query]"),
        ("[sparql]", "[/sparql]", "[sparql]", "[/sparql]"),
        ("<query>", "</query>", "<query>", "</query>"),
        ("<sparql>", "</sparql>", "<sparql>", "</sparql>"),
        ("`sparql\n", "`", "`sparql\n", "`"),
        ("`sparql\\n", "`", "`sparql\n", "`"),
    ],
    name_func=_get_name_for_get_args)
    def test_parse_args_testing_start_end_tokens(self, start_tag, end_tag, expected_start_tag, expected_end_tag):
        list_args = [
            "--train-data", "dont care",
            "--start-tag", start_tag,
            "--end-tag", end_tag,
            ]
        
        args = parse_args(list_args)
        
        self.assertEqual(args.start_tag, expected_start_tag)
        self.assertEqual(args.end_tag, expected_end_tag)
    
    @parameterized.expand([
        ("[query]", "[/query]"),
        ("[sparql]", "[/sparql]"),
        ("<query>", "</query>"),
        ("<sparql>", "</sparql>"),
        ("`sparql\n", "`")
    ],
    name_func=_get_name_for_get_args)
    def test_extract_query_correct_form_1(self, start_tag, end_tag):
        query = f"{start_tag}the query{end_tag}"
        expected_query = "the query"
        
        scripts.sft_peft.start_tag = start_tag
        scripts.sft_peft.end_tag = end_tag
        
        self.assertEqual(expected_query, extract_query(query))
    
    @parameterized.expand([
        ("[query]", "[/query]"),
        ("[sparql]", "[/sparql]"),
        ("<query>", "</query>"),
        ("<sparql>", "</sparql>"),
        ("`sparql\n", "`")
    ],
    name_func=_get_name_for_get_args)
    def test_extract_query_correct_form_2(self, start_tag, end_tag):
        query = f"IRRELEVANT{start_tag}the query{end_tag}"
        expected_query = "the query"
        
        scripts.sft_peft.start_tag = start_tag
        scripts.sft_peft.end_tag = end_tag
        
        self.assertEqual(expected_query, extract_query(query))
        
    @parameterized.expand([
        ("[query]", "[/query]"),
        ("[sparql]", "[/sparql]"),
        ("<query>", "</query>"),
        ("<sparql>", "</sparql>"),
        ("`sparql\n", "`")
    ],
    name_func=_get_name_for_get_args)
    def test_extract_query_correct_form_3(self, start_tag, end_tag):
        query = f"{start_tag}the query{end_tag}IRRELEVANT"
        expected_query = "the query"
        
        scripts.sft_peft.start_tag = start_tag
        scripts.sft_peft.end_tag = end_tag
        
        self.assertEqual(expected_query, extract_query(query))
        
    @parameterized.expand([
        ("[query]", "[/query]"),
        ("[sparql]", "[/sparql]"),
        ("<query>", "</query>"),
        ("<sparql>", "</sparql>"),
        ("`sparql\n", "`")
    ],
    name_func=_get_name_for_get_args)
    def test_extract_query_correct_form_4(self, start_tag, end_tag):
        query = f"IRRELEVANT{start_tag}the query{end_tag}IRRELEVANT"
        expected_query = "the query"
        
        scripts.sft_peft.start_tag = start_tag
        scripts.sft_peft.end_tag = end_tag
        
        self.assertEqual(expected_query, extract_query(query))
    
    @parameterized.expand([
        ("[query]", "[/query]"),
        ("[sparql]", "[/sparql]"),
        ("<query>", "</query>"),
        ("<sparql>", "</sparql>"),
        ("`sparql\n", "`")
    ],
    name_func=_get_name_for_get_args)
    def test_extract_query_not_correct_form_1(self, start_tag, end_tag):
        query = f"the query{end_tag}"
        expected_query = None
        
        scripts.sft_peft.start_tag = start_tag
        scripts.sft_peft.end_tag = end_tag
        
        self.assertEqual(expected_query, extract_query(query))
    
    @parameterized.expand([
        ("[query]", "[/query]"),
        ("[sparql]", "[/sparql]"),
        ("<query>", "</query>"),
        ("<sparql>", "</sparql>"),
        ("`sparql\n", "`")
    ],
    name_func=_get_name_for_get_args)
    def test_extract_query_not_correct_form_2(self, start_tag, end_tag):
        query = f"{start_tag}the query"
        expected_query = None
        
        scripts.sft_peft.start_tag = start_tag
        scripts.sft_peft.end_tag = end_tag
        
        self.assertEqual(expected_query, extract_query(query))
    
    @parameterized.expand([
        ("[query]", "[/query]"),
        ("[sparql]", "[/sparql]"),
        ("<query>", "</query>"),
        ("<sparql>", "</sparql>"),
        ("`sparql\n", "`")
    ],
    name_func=_get_name_for_get_args)
    def test_extract_query_not_correct_form_3(self, start_tag, end_tag):
        query = f"the query"
        expected_query = None
        
        scripts.sft_peft.start_tag = start_tag
        scripts.sft_peft.end_tag = end_tag
        
        self.assertEqual(expected_query, extract_query(query))