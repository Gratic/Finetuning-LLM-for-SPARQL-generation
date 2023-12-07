from .EntityExtractor import LLMEntityExtractor
from .EntityLinker import FirstWikidataEntityLinker
from .LLMConnector import LlamaCPPConnector
from .Pipeline import OrderedPipeline
from .PipelineFeeder import SimplePipelineFeeder
from .PlaceholderFiller import SimplePlaceholderFiller
from .TemplateLLMQuerySender import TemplateLLMQuerySender, BASE_TEMPLATE
from .Translator import LLMTranslator
from .SentencePlaceholder import SimpleSentencePlaceholder

if __name__ == "__main__":
    llm_connector = LlamaCPPConnector()
    templateLLMQuerySender = TemplateLLMQuerySender(llm_connector, BASE_TEMPLATE, '[', ']')
    
    pipeline = OrderedPipeline()
    pipeline.add_step(LLMEntityExtractor(templateLLMQuerySender))
    pipeline.add_step(SimpleSentencePlaceholder())
    pipeline.add_step(FirstWikidataEntityLinker())
    pipeline.add_step(LLMTranslator(templateLLMQuerySender))
    pipeline.add_step(SimplePlaceholderFiller())
    
    # pipeline.add_step(LLMTranslator(templateLLMQuerySender))
    
    
    feeder = SimplePipelineFeeder(pipeline)
    results = feeder.process(["How many countries in the EU?"])
    
    print(results)