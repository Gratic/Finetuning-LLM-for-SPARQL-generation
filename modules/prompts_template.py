import sys
from pathlib import Path
sys.path.append(Path("modules").absolute().__str__())

BASE_SYSTEM_PROMPT = """This is a conversation between User and Llama, a friendly chatbot. Llama is helpful, kind, honest, good at writing, and never fails to answer any requests immediately and with precision.
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

BASE_INSTRUCTION = """Instructions:
Given a question, generate a SPARQL query that answers the question. The SPARQL query must work with the knowledge graph of Wikidata. Don't add extra comments. Sandwich the SPARQL query between character backquotes like "`".
Strictly apply these instructions using this sentence: """

BASE_BASIC_INSTRUCTION = """Given a question, generate a SPARQL query that answers the question. Answer this following question or instruction:"""

PERSONA_BASIC_INSTRUCTION = """You're a SPARQL assistant. You're an expert at SPARQL and all your answers are SPARQL only. You were given the following question or instruction, please answer a SPARQL query that can answer it:"""

ELABORATE_INSTRUCTION = """Your assignment involves a two-step process:
First, you will receive a task described in a single sentence. This task outlines what information you need to find or what question you need to answer. Read it properly.
Then, you will create a SPARQL Query: Based on the task given, you will write a SPARQL query. SPARQL is a specialized query language used to retrieve and manipulate data stored in Resource Description Framework (RDF) format. Your goal is to craft a query that, when executed, fetches the data or answers required by the initial instruction.
Make sure your SPARQL query is correctly formulated so that, upon execution, it produces the desired result matching the task's requirements.
Answer this following instruction:"""

BASE_ANNOTATED_INSTRUCTION_ONE_SHOT = """Given a question, generate a SPARQL query where entities and properties are placeholders that answers the question. After the generated query, gives the list of placeholders and there corresponding Wikidata identifiers. Don't add extra comments.

Example 0:
Can you tell me how many scientific articles there are on Wikidata?
`sparql
SELECT (COUNT(?article) AS ?count)
WHERE {?article [property 0]/[property 1] [entity 0]}
`
[entity 0] -> "wd:Q13442814"
[property 0] -> "wdt:P31"
[property 1] -> "wdt:P279"
End of Example 0.

Strictly apply these instructions using this sentence: """

### TEMPLATES

LLAMA2_TEMPLATE = """[INST] <<SYS>>
[system_prompt]
<</SYS>>

[prompt] [/INST] """

BASE_MISTRAL_TEMPLATE = """[INST] [system_prompt] [prompt] [/INST] """

CODELLAMA_TEMPLATE = """[INST] [system_prompt]
[prompt]
[/INST] """

LLAMA3_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

[system_prompt]<|eot_id|><|start_header_id|>user<|end_header_id|>

[prompt]<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

def get_template_for_model(model_id: str) -> str:
    if not isinstance(model_id, str):
        raise TypeError("model_id must be a str.")
    
    template = None
    model_id_lower = model_id.lower()
    if "llama-2" in model_id_lower:
        template = LLAMA2_TEMPLATE
    elif "llama-3" in model_id_lower:
        template = LLAMA3_TEMPLATE
    elif "codellama" in model_id_lower:
        template = CODELLAMA_TEMPLATE
    elif "mistral" in model_id_lower:
        template = BASE_MISTRAL_TEMPLATE
        
    if template == None:
        raise NotImplementedError("There is no appropriate template for this model yet.")
    
    return template