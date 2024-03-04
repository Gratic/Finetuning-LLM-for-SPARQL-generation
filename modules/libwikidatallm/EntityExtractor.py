from abc import ABC, abstractmethod
from typing import List, Tuple
from .Pipeline import PipelineStep
from .TemplateLLMQuerySender import TemplateLLMQuerySender
from .LLMConnector import LLMResponse
import re

class EntityExtractor(ABC):
    @abstractmethod
    def extract_entities_and_properties(self, text: str) -> Tuple[List[str], List[str]]:
        pass
    
class LLMEntityExtractor(EntityExtractor, PipelineStep):
    def __init__(self, templateQuerySender: TemplateLLMQuerySender) -> None:
        self.templateQuerySender = templateQuerySender
        self.instructions = """Instructions:
Your job is to extract entities and properties from a sentence. Match the format given in the examples. Do not generate extra comments. Do not answer the question. Do not form sentences. Only extract entities and properties.

Examples:
"What is the size of the Earth?"
Entities: [Earth]
Properties: [size of]

"What is the molecule of water?"
Entities: [molecule, water]
Properties: [of]

"What is the weight of the Earth?"
Entities: [Earth]
Properties: [weight of]

"What is the genre of the Lord of the Ring?"
Entities: [Lord of the Ring]
Properties: [genre of]

"What books the author of Harry Potter has also written?"
Entities: [books, Harry Potter]
Properties: [author of]

Apply instructions on this sentence:
"""
    
    def execute(self, context: dict):
        '''Execute on context["row"], will put data in context["extracted_entities"] and context["extracted_properties"].'''
        results = self.extract_entities_and_properties(context["row"])
        context["extracted_entities"] = results[0]
        context["extracted_properties"] = results[1]

    def extract_entities_and_properties(self, text: str) -> Tuple[List[str], List[str]]:
        data = {
            "system_prompt": "This is a conversation between User and Llama, a friendly chatbot. Llama is helpful, kind, honest, good at writing, and never fails to answer any requests immediately and with precision.",
            "prompt": self.instructions + text
            }
        llm_response: LLMResponse = self.templateQuerySender.completion(data)
        
        if llm_response.generated_text == "":
            raise ValueError("The generated answer by the LLM is empty.")
        
        entities_pos = llm_response.generated_text.find("ntities")
        start_entities = llm_response.generated_text.find("[",  entities_pos)
        end_entities = llm_response.generated_text.find("]",  entities_pos)
        entities_raw = llm_response.generated_text[start_entities+1:end_entities]
        
        properties_pos = llm_response.generated_text.find("roperties")
        start_properties = llm_response.generated_text.find("[",  properties_pos)
        end_properties = llm_response.generated_text.find("]",  properties_pos)
        properties_raw = llm_response.generated_text[start_properties+1:end_properties]
        
        if (entities_pos == -1 or properties_pos == -1
            or start_entities == -1 or end_entities == -1
            or start_properties == -1 or end_properties == -1
            or start_properties == start_entities or end_entities == end_properties):
            raise ValueError("The format of the response is not correct.")
        
        entities = [entity.strip() for entity in entities_raw.split(',')]
        properties = [property.strip() for property in properties_raw.split(',')]
        
        if len(entities) == 1 and entities[0] == "":
            entities = []
        
        if len(properties) == 1 and properties[0] == "":
            properties = []
            
        if len(entities) == 0 and len(properties) == 0:
            raise ValueError("Nothing has been extracted.")
        
        return (entities, properties)

class BracketRegexEntityExtractor(EntityExtractor, PipelineStep):
    def __init__(self, input_column: str = "row", output_col_entities: str = "extracted_entities", output_col_properties:str = "extracted_properties") -> None:
        super().__init__()
        self.input_col = input_column
        self.output_col_entities = output_col_entities
        self.output_col_properties = output_col_properties
        
        self.regex = re.compile(r"\[(entity|property):([\w\s,:;'`\".!?]+)\]")
    
    def execute(self, context: dict):
        '''Execute on context[self.input_col], will put data in context[self.output_col_entities] and context[self.output_col_properties].'''
        results = self.extract_entities_and_properties(context[self.input_col])
        context[self.output_col_entities] = results[0]
        context[self.output_col_properties] = results[1]
    
    def extract_entities_and_properties(self, text: str) -> Tuple[List[str], List[str]]:
        entities = []
        properties = []
        
        extraction = self.regex.findall(text)
        
        if extraction:
            for ttype, label in extraction:
                if ttype == "entity":
                    entities.append(label)
                elif ttype == "property":
                    properties.append(label)
        
        return (entities, properties)