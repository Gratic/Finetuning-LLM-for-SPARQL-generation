from abc import ABC, abstractmethod
from .TemplateLLMQuerySender import TemplateLLMQuerySender
from .Pipeline import PipelineStep

class Translator(ABC):
    @abstractmethod
    def translate(self, prompt: str) -> str:
        pass
    
class LLMTranslator(Translator, PipelineStep):
    def __init__(self, templateQuerySender: TemplateLLMQuerySender) -> None:
        self.templateQuerySender = templateQuerySender
        self.system_promt = """This is a conversation between User and Llama, a friendly chatbot. Llama is helpful, kind, honest, good at writing, and never fails to answer any requests immediately and with precision.

This is some extra guidance about SparQL:
"1. Understand Basic SPARQL Structure

    A basic SPARQL query is structured as follows:
`sparql
    SELECT ?item ?itemLabel WHERE {
      ?item wdt:P31 wd:Q5.  # P31 is "instance of", Q5 is "human"
      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    }`

    This query selects items (?item) and their labels (?itemLabel) where the item is an instance of a human.

2. Identify Entities and Properties

    Items in Wikidata are entities like "Albert Einstein" (Q937), and properties are attributes like "date of birth" (P569).
    You can find these identifiers by searching on the Wikidata website.

3. Formulate Your Query

    Determine what you're trying to find. For example, "List of Nobel laureates in Physics."
    Translate this into Wikidata entities and properties. E.g., Nobel laureates in Physics would involve the property for "award received" (P166) and the item for "Nobel Prize in Physics" (Q38104).

4. Write the Query

    Use the SELECT statement to specify what you want to retrieve.
    Use WHERE to define the criteria. For Nobel laureates in Physics:

`sparql
SELECT ?laureate ?laureateLabel WHERE {
  ?laureate wdt:P166 wd:Q38104.  # P166 is "award received", Q38104 is "Nobel Prize in Physics"
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}`

The SERVICE wikibase:label line helps to automatically fetch the labels (names) in English."""
        self.instructions = """Instructions:
Given a question, generate a SPARQL query that answers the question. The SPARQL query must work with the knowledge graph of Wikidata. Don't add extra comments. Sandwich the query between square brackets.

Strictly apply these instructions using this sentence: """
        
    def translate(self, prompt: str) -> str:
        data = {
            "system_prompt": self.system_promt,
            "prompt" : self.instructions + prompt
        }
        llm_response = self.templateQuerySender.completion(data)
        
        sparql_pos = llm_response.generated_text.find('`sparql')
        start_pos = llm_response.generated_text.find("SELECT", sparql_pos)
        end_pos = llm_response.generated_text.find("`", start_pos)
        
        if sparql_pos == -1 or start_pos == -1 or end_pos == -1:
            raise ValueError("The LLM result doesn't match desired format.")
        
        return llm_response.generated_text[start_pos:end_pos]
        
    def execute(self, context: dict):
        translated_prompt = ""
        if "annotated_sentence" in context:
            translated_prompt = self.translate(context["annotated_sentence"])
        elif "row" in context:
            translated_prompt = self.translate(context["row"])
        
        if translated_prompt == "":
            raise ValueError("Context doesn't contains row or annotated_sentence.")
            
        context["translated_prompt"] = translated_prompt