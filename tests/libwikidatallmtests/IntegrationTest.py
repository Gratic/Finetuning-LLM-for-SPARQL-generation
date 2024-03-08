from typing import List
import unittest
from modules.libwikidatallm.__main__ import template_pipeline
import pandas as pd
from modules.libwikidatallm.PipelineFeeder import SimplePipelineFeeder
from modules.libwikidatallm.LLMConnector import LLMConnector, LLMResponse
from modules.prompts_template import BASE_MISTRAL_TEMPLATE

class MockLLMConnector(LLMConnector):
    def __init__(self, query:str) -> None:
        super().__init__()
        self.query = query
        
    def completion(self, prompt: str) -> LLMResponse:
        return LLMResponse(self.query, self.query)
    
    def tokenize(self, prompt: str) -> List[int]:
        return len(self.query.split())

class IntegrationTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        
    def test_template_pipeline_working_templated_query(self):
        query = """`sparql
SELECT ?l ?lemma WHERE {
?l dct:language wd:[entity:Danish];
wikibase:lemma ?lemma;
wikibase:lexicalCategory wd:[entity:idiom].
}
`"""
        dataset = ["a prompt, this is insignificant here because the query generated is rigged with MockLLMConnector"]
        pipeline = template_pipeline(llm_connector=MockLLMConnector(query), template=BASE_MISTRAL_TEMPLATE)
        feeder = SimplePipelineFeeder(pipeline)
        
        resulting_query = """SELECT ?l ?lemma WHERE {
?l dct:language wd:Q9035;
wikibase:lemma ?lemma;
wikibase:lexicalCategory wd:Q184511.
}"""
        
        results = feeder.process(dataset)
        
        self.assertEqual(1, len(results))
        
        result = results[0]
        self.assertTrue('row' in result)
        self.assertTrue('translated_prompt' in result)
        self.assertTrue('extracted_entities' in result)
        self.assertTrue('extracted_properties' in result)
        self.assertTrue('linked_entities' in result)
        self.assertTrue('linked_properties' in result)
        self.assertTrue('output' in result)
        self.assertTrue('last_executed_step' in result)
        self.assertTrue('to_be_executed_step' in result)
        self.assertTrue('status' in result)
        self.assertTrue('has_error' in result)
        
        self.assertFalse(result['has_error'])
        
        self.assertEqual(resulting_query, result['output'])
        
    def test_template_pipeline_working_no_placeholder_query(self):
        query = """`sparql
SELECT ?property ?propertyType ?propertyLabel ?propertyDescription WHERE {
?property wikibase:propertyType ?propertyType .
SERVICE wikibase:label { bd:serviceParam wikibase:language \"[AUTO_LANGUAGE],en\". }
} ORDER BY ASC(xsd:integer(STRAFTER(STR(?property), 'P')))
`"""
        dataset = ["a prompt, this is insignificant here because the query generated is rigged with MockLLMConnector"]
        pipeline = template_pipeline(llm_connector=MockLLMConnector(query), template=BASE_MISTRAL_TEMPLATE)
        feeder = SimplePipelineFeeder(pipeline)
        
        resulting_query = """SELECT ?property ?propertyType ?propertyLabel ?propertyDescription WHERE {
?property wikibase:propertyType ?propertyType .
SERVICE wikibase:label { bd:serviceParam wikibase:language \"[AUTO_LANGUAGE],en\". }
} ORDER BY ASC(xsd:integer(STRAFTER(STR(?property), 'P')))"""
        
        results = feeder.process(dataset)
        
        self.assertEqual(1, len(results))
        
        result = results[0]
        self.assertTrue('row' in result)
        self.assertTrue('translated_prompt' in result)
        self.assertTrue('extracted_entities' in result)
        self.assertTrue('extracted_properties' in result)
        self.assertTrue('linked_entities' in result)
        self.assertTrue('linked_properties' in result)
        self.assertTrue('output' in result)
        self.assertTrue('last_executed_step' in result)
        self.assertTrue('to_be_executed_step' in result)
        self.assertTrue('status' in result)
        self.assertTrue('has_error' in result)
        
        self.assertFalse(result['has_error'])
        
        self.assertEqual(resulting_query, result['output'])