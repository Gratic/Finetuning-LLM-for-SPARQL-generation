from typing import List
import unittest
from modules.libwikidatallm.TemplateLLMQuerySender import TemplateLLMQuerySender
from modules.libwikidatallm.LLMConnector import LLMConnector, LLMResponse

class LLMConnectorDummy(LLMConnector):
    def completion(self, prompt: str) -> LLMResponse:
        return LLMResponse(prompt, prompt)
    
    def tokenize(self, prompt: str) -> List[int]:
        return [len(word) for word in prompt]

class TemplateLLMQuerySenderTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        
    def test_initialization(self):
        llm = LLMConnectorDummy()
        template = "template_test"
        tqs = TemplateLLMQuerySender(llm, template)
        
        self.assertIs(tqs.llm, llm)
        self.assertEqual(tqs.template_text, template)
        self.assertEqual(tqs.start_seq, "")
        self.assertEqual(tqs.end_seq, "")
        
    def test_apply_template_normal_entry_no_seq(self):
        data = {
            "user": "bot",
            "food": "an apple"
        }
        
        llm = LLMConnectorDummy()
        template = "The user is eating food."
        tqs = TemplateLLMQuerySender(llm, template)
        
        self.assertEqual(tqs.apply_template(data), "The bot is eating an apple.")
        
    def test_apply_template_normal_entry_brackets_seq(self):
        data = {
            "user": "bot",
            "food": "an apple"
        }
        
        llm = LLMConnectorDummy()
        template = "The [user] is eating [food]."
        tqs = TemplateLLMQuerySender(llm, template, "[", "]")
        
        self.assertEqual(tqs.apply_template(data), "The bot is eating an apple.")

    def test_apply_template_wrong_entry_type(self):
        data = ["bot"]
        
        llm = LLMConnectorDummy()
        template = "The user is eating food."
        tqs = TemplateLLMQuerySender(llm, template)
        
        with self.assertRaises(TypeError):
            tqs.apply_template(data)
            
    def test_apply_template_wrong_dict_key_is_not_str(self):
        data = {
            0: "bot"
        }
        
        llm = LLMConnectorDummy()
        template = "The 0 is eating food."
        tqs = TemplateLLMQuerySender(llm, template)
        
        with self.assertRaises(TypeError):
            tqs.apply_template(data)
    
    def test_apply_template_good_dict_values_is_not_str(self):
        data = {
            "user": 0
        }
        
        llm = LLMConnectorDummy()
        template = "The user is eating food."
        tqs = TemplateLLMQuerySender(llm, template)
        
        self.assertEqual(tqs.apply_template(data), "The 0 is eating food.")