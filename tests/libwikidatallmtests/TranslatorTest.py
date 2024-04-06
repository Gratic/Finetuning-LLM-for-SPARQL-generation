import unittest
from modules.libwikidatallm.Translator import LLMTranslator
from modules.libwikidatallm.LLMConnector import LLMResponse
from modules.libwikidatallm.Pipeline import NoSparqlMatchError
from parameterized import parameterized
from itertools import product

def _get_name_for_get_args(testcase_func, param_num, param):
    return f"{testcase_func.__name__}_{param_num}_{param[0][0]}"

class MockTemplateQuerySender():
    def __init__(self, id, start_tag, end_tag) -> None:
        self.id = id
        self.start_tag = start_tag
        self.end_tag = end_tag
    
    def completion(self, data):
        if self.id == 0: return LLMResponse("", f"{self.start_tag}SELECT{self.end_tag}")
        if self.id == 1: return LLMResponse("", f"{self.start_tag}{self.end_tag}")
        if self.id == 2: return LLMResponse("", f"{self.start_tag}PREFIX prefix\nSELECT select{self.end_tag}")
        if self.id == 3: return LLMResponse("", f"IRRELEVANT{self.start_tag}SELECT{self.end_tag}")
        if self.id == 4: return LLMResponse("", f"IRRELEVANT{self.start_tag}{self.end_tag}")
        if self.id == 5: return LLMResponse("", f"IRRELEVANT{self.start_tag}PREFIX prefix\nSELECT select{self.end_tag}")
        if self.id == 6: return LLMResponse("", f"{self.start_tag}SELECT{self.end_tag}IRRELEVANT")
        if self.id == 7: return LLMResponse("", f"{self.start_tag}{self.end_tag}IRRELEVANT")
        if self.id == 8: return LLMResponse("", f"{self.start_tag}PREFIX prefix\nSELECT select{self.end_tag}IRRELEVANT")
        if self.id == 9: return LLMResponse("", f"IRRELEVANT{self.start_tag}SELECT{self.end_tag}IRRELEVANT")
        if self.id == 10: return LLMResponse("", f"IRRELEVANT{self.start_tag}{self.end_tag}IRRELEVANT")
        if self.id == 11: return LLMResponse("", f"IRRELEVANT{self.start_tag}PREFIX prefix\nSELECT select{self.end_tag}IRRELEVANT")
    
    def expected(self):
        if self.id == 0: return "SELECT"
        if self.id == 2: return "PREFIX prefix\nSELECT select"
        if self.id == 3: return "SELECT"
        if self.id == 5: return "PREFIX prefix\nSELECT select"
        if self.id == 6: return "SELECT"
        if self.id == 8: return "PREFIX prefix\nSELECT select"
        if self.id == 9: return "SELECT"
        if self.id == 11: return "PREFIX prefix\nSELECT select"
    
    def expect_throw(self):
        if self.id == 1: return True
        if self.id == 4: return True
        if self.id == 7: return True
        if self.id == 10: return True
        return False

class LLMTranslatorTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
    
    
    
    @parameterized.expand(list(product(list(range(12)), [
        ("[query]", "[/query]"),
        ("[sparql]", "[/sparql]"),
        ("<query>", "</query>"),
        ("`sparql\n", "`")
    ])),
    name_func=_get_name_for_get_args)
    def test_translate_(self, id, tuple_tag):
        start_tag, end_tag = tuple_tag
        
        templater = MockTemplateQuerySender(
            id=id,
            start_tag=start_tag,
            end_tag=end_tag
        )
        
        translater = LLMTranslator(
            templateQuerySender=templater,
            system_prompt="",
            instruction_prompt="",
            start_tag=start_tag,
            end_tag=end_tag,
        )
        
            
        if not templater.expect_throw():
            translation = translater.translate("doesn't matter")
            self.assertEqual(translation, templater.expected())
        elif templater.expect_throw():
            self.assertRaises(NoSparqlMatchError, translater.translate, "doesnt matter")
        