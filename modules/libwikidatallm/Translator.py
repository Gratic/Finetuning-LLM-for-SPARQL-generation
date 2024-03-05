from .Pipeline import PipelineStep, NoSparqlMatchError
from .TemplateLLMQuerySender import TemplateLLMQuerySender
from abc import ABC, abstractmethod
from prompts_template import BASE_SYSTEM_PROMPT, BASE_BASIC_INSTRUCTION

class Translator(ABC):
    @abstractmethod
    def translate(self, prompt: str) -> str:
        pass
    
class LLMTranslator(Translator, PipelineStep):
    def __init__(self, templateQuerySender: TemplateLLMQuerySender, system_prompt: str = BASE_SYSTEM_PROMPT, instruction_prompt: str = BASE_BASIC_INSTRUCTION, input_column:str = 'row', output_column:str = 'translated_prompt') -> None:
        self.templateQuerySender = templateQuerySender
        self.system_prompt = system_prompt
        self.instructions = instruction_prompt
        self.last_response = None
        self.input_column = input_column
        self.output_column = output_column
        
    def translate(self, question: str) -> str:
        data = {
            "system_prompt": self.system_prompt,
            "prompt" : self.instructions + question
        }
        llm_response = self.templateQuerySender.completion(data)
        self.last_response = llm_response
        
        sparql_pos = llm_response.generated_text.find('`sparql')
        start_pos = llm_response.generated_text.find("SELECT", sparql_pos)
        end_pos = llm_response.generated_text.find("`", start_pos)
        
        if sparql_pos == -1 or start_pos == -1 or end_pos == -1:
            raise NoSparqlMatchError(msg="The LLM result doesn't match desired format.", sparql=llm_response.generated_text)
        
        return llm_response.generated_text[start_pos:end_pos].strip()
        
    def execute(self, context: dict):
        try:
            translated_prompt = self.translate(context[self.input_column])
            
            if translated_prompt == "":
                raise ValueError("Context doesn't contains row or annotated_sentence.")
                
            context[self.output_column] = translated_prompt
        except NoSparqlMatchError as exception:
            context[self.output_column] = ""
            raise exception