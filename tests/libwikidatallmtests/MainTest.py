import unittest
from modules.libwikidatallm.__main__ import get_args
from parameterized import parameterized
def _get_name_for_get_args(testcase_func, param_num, param):
    return f"{testcase_func.__name__}_{param_num}"
        

class MainTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
    
    
    @parameterized.expand([
        ("[query]", "[/query]", "[query]", "[/query]"),
        ("`sparql\n", "`", "`sparql\n", "`"),
        ("`sparql\\n", "`", "`sparql\n", "`"),
        ("[sparql]", "[/sparql]", "[sparql]", "[/sparql]"),
        ("<query>", "</query>", "<query>", "</query>"),
        ("<sparql>", "</sparql>", "<sparql>", "</sparql>"),
    ],
    name_func=_get_name_for_get_args)
    def test_get_args(self, start_tag, end_tag, expected_start_tag, expected_end_tag):
        list_args = [
            "--model", "a model",
            "--tokenizer", "a tokenizer",
            "--start-tag", start_tag,
            "--end-tag", end_tag,
            ]
        
        args = get_args(list_args)
        
        self.assertEqual(args.start_tag, expected_start_tag)
        self.assertEqual(args.end_tag, expected_end_tag)