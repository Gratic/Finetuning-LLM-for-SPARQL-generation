from typing import List
import unittest
from modules.libwikidatallm.__main__ import template_pipeline
import pandas as pd
from modules.libwikidatallm.PipelineFeeder import SimplePipelineFeeder
from modules.libwikidatallm.LLMConnector import LLMConnector, LLMResponse
from modules.prompts_template import BASE_MISTRAL_TEMPLATE

class MockLLMConnector(LLMConnector):
    def __init__(self) -> None:
        super().__init__()
        self.query = """`sparql
SELECT ?l ?lemma WHERE {
?l dct:language wd:[entity:Danish];
wikibase:lemma ?lemma;
wikibase:lexicalCategory wd:[entity:idiom].
}
`"""
        
    def completion(self, prompt: str) -> LLMResponse:
        return LLMResponse(self.query, self.query)
    
    def tokenize(self, prompt: str) -> List[int]:
        return len(self.query.split())

class IntegrationTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        
    def test_template_pipeline_working_templated_query(self):
        dataset = ["a prompt, this is insignificant here because the query generated is rigged with MockLLMConnector"]
        pipeline = template_pipeline(llm_connector=MockLLMConnector(), template=BASE_MISTRAL_TEMPLATE)
        feeder = SimplePipelineFeeder(pipeline)
        
        resulting_query = """SELECT ?l ?lemma WHERE {
?l dct:language wd:Q9035;
wikibase:lemma ?lemma;
wikibase:lexicalCategory wd:Q184511.
}"""
        
        results = feeder.process(dataset)
        
        print("STUFF")
        
        self.assertEqual(1, len(results))
        
        result = results[0]
        self.assertTrue('row' in result)
        self.assertTrue('translated_prompt' in result)
        self.assertTrue('extracted_entities' in result)
        self.assertTrue('extracted_properties' in result)
        self.assertTrue('linked_entities' in result)
        self.assertTrue('linked_properties' in result)
        self.assertTrue('linked_query' in result)
        self.assertTrue('last_executed_step' in result)
        self.assertTrue('to_be_executed_step' in result)
        self.assertTrue('status' in result)
        self.assertTrue('has_error' in result)
        
        self.assertFalse(result['has_error'])
        
        print(resulting_query)
        print(result['linked_query'])
        
        self.assertEqual(resulting_query, result['linked_query'])