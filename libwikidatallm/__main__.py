from .EntityExtractor import LLMEntityExtractor
from .EntityLinker import FirstWikidataEntityLinker
from .LLMConnector import LlamaCPPConnector, vLLMConnector
from .Pipeline import OrderedPipeline
from .PipelineFeeder import SimplePipelineFeeder
from .PlaceholderFiller import SimplePlaceholderFiller
from .TemplateLLMQuerySender import TemplateLLMQuerySender, BASE_LLAMA_TEMPLATE, BASE_MISTRAL_TEMPLATE
from .Translator import LLMTranslator, BASE_ANNOTATED_INSTRUCTION, BASE_ANNOTATED_INSTRUCTION_ONE_SHOT
from .SentencePlaceholder import SimpleSentencePlaceholder

if __name__ == "__main__":
    # llm_connector = LlamaCPPConnector()
    llm_connector = vLLMConnector(model_path="outputs/merged_model/Mistral-7B-Instruct-v0.2-merged",
                                  tokenizer="mistralai/Mistral-7B-Instruct-v0.2",
                                  context_length=4096)
    pipeline = OrderedPipeline()
    
    # Template pipeline
    # templateLLMQuerySender = TemplateLLMQuerySender(llm_connector, BASE_LLAMA_TEMPLATE, '[', ']')
    # pipeline.add_step(LLMEntityExtractor(templateLLMQuerySender))
    # pipeline.add_step(SimpleSentencePlaceholder())
    # pipeline.add_step(FirstWikidataEntityLinker())
    # pipeline.add_step(LLMTranslator(templateLLMQuerySender))
    # pipeline.add_step(SimplePlaceholderFiller())
    
    # Only LLM pipeline
    # templateLLMQuerySender = TemplateLLMQuerySender(llm_connector, BASE_LLAMA_TEMPLATE, '[', ']')
    # pipeline.add_step(LLMTranslator(templateLLMQuerySender))
    
    # LLM create annotated sparql and link the placeholders
    templateLLMQuerySender = TemplateLLMQuerySender(llm_connector, BASE_MISTRAL_TEMPLATE, "[", "]")
    pipeline.add_step(LLMTranslator(templateLLMQuerySender, "", BASE_ANNOTATED_INSTRUCTION_ONE_SHOT))
    
    feeder = SimplePipelineFeeder(pipeline)
    results = feeder.process(["How many countries in the EU?"])
    
    print(results)