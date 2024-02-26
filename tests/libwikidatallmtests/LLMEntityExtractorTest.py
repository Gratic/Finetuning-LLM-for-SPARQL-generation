import unittest
from modules.libwikidatallm.EntityExtractor import LLMEntityExtractor
from modules.libwikidatallm.LLMConnector import LLMResponse
from modules.libwikidatallm.TemplateLLMQuerySender import TemplateLLMQuerySender
from typing import Dict

class TemplateLLMQuerySenderDummy(TemplateLLMQuerySender):
    def __init__(self, test_case: int = 0) -> None:
        self.test_case = test_case
    
    def completion(self, data: Dict[str, str]) -> LLMResponse:
        message = ""
        if self.test_case == 0:
            message = "Entities: [director, Memento]\nProperties: [other films of]"
        elif self.test_case == 1:
            message = "Entities: [a]\nProperties: []"
        elif self.test_case == 2:
            message = "Entities: []\nProperties: [a]"
        elif self.test_case == 3:
            message = "Properties: [c, d]\nEntities: [a, b]"
        elif self.test_case == 4:
            message = "Properties: []\nEntities: []"
        elif self.test_case == 5:
            message = ""
        elif self.test_case == 6:
            message = "This response is obviously not in the good format."
        elif self.test_case == 7:
            message = "This response contains Entities but not the other."
        elif self.test_case == 8:
            message = "This response contains Properties but not the other."
        elif self.test_case == 9:
            message = "This response contains Entities and Properties but is still not in the good format."
        return LLMResponse(message, message)
        

class LLMEntityExtractorTest(unittest.TestCase):
    
    def test_extraction_good_format(self):
        ee = LLMEntityExtractor(TemplateLLMQuerySenderDummy())
        
        entities, properties = ee.extract_entities_and_properties("test_case normal")
        self.assertEqual(entities, ["director", "Memento"])
        self.assertEqual(properties, ["other films of"])
        
    def test_extraction_good_format_no_properties(self):
        ee = LLMEntityExtractor(TemplateLLMQuerySenderDummy(1))
        
        entities, properties = ee.extract_entities_and_properties("test_case normal")
        self.assertEqual(entities, ["a"])
        self.assertEqual(properties, [])
        
    def test_extraction_good_format_no_entities(self):
        ee = LLMEntityExtractor(TemplateLLMQuerySenderDummy(2))
        
        entities, properties = ee.extract_entities_and_properties("test_case normal")
        self.assertEqual(entities, [])
        self.assertEqual(properties, ["a"])
        
    def test_extraction_reverse_format(self):
        ee = LLMEntityExtractor(TemplateLLMQuerySenderDummy(3))
        
        entities, properties = ee.extract_entities_and_properties("test_case normal")
        self.assertEqual(entities, ["a", "b"])
        self.assertEqual(properties, ["c", "d"])
        
    def test_extraction_both_empty(self):
        ee = LLMEntityExtractor(TemplateLLMQuerySenderDummy(4))
        
        with self.assertRaises(ValueError):
            ee.extract_entities_and_properties("test_case fautif")
        
    def test_extraction_response_empty(self):
        ee = LLMEntityExtractor(TemplateLLMQuerySenderDummy(5))
        
        with self.assertRaises(ValueError):
            ee.extract_entities_and_properties("test_case fautif")
        
    def test_extraction_response_has_no_entities_and_properties(self):
        ee = LLMEntityExtractor(TemplateLLMQuerySenderDummy(6))
        
        with self.assertRaises(ValueError):
            ee.extract_entities_and_properties("test_case fautif")
        
    def test_extraction_response_has_no_properties(self):
        ee = LLMEntityExtractor(TemplateLLMQuerySenderDummy(7))
        
        with self.assertRaises(ValueError):
            ee.extract_entities_and_properties("test_case fautif")
        
    def test_extraction_response_has_no_entities(self):
        ee = LLMEntityExtractor(TemplateLLMQuerySenderDummy(8))
        
        with self.assertRaises(ValueError):
            ee.extract_entities_and_properties("test_case fautif")
        
    def test_extraction_response_has_both_but_both_are_empty(self):
        ee = LLMEntityExtractor(TemplateLLMQuerySenderDummy(9))
        
        with self.assertRaises(ValueError):
            ee.extract_entities_and_properties("test_case fautif")