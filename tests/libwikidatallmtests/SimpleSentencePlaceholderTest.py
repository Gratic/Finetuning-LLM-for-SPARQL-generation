import unittest
from modules.libwikidatallm.SentencePlaceholder import SimpleSentencePlaceholder

class SimpleSentencePlaceholderTest(unittest.TestCase):
    
    def test_annotate_correct_data_one_of_both(self):
        message = "What is the circumference of the Earth?"
        entities = ["Earth"]
        properties = ["circumference of"]
        
        annotator = SimpleSentencePlaceholder()
        self.assertEqual(annotator.annotate(message, entities, properties), "What is the [property 0] the [entity 0]?")
    
    def test_annotate_correct_data_two_of_entities(self):
        message = "What are the books of Harry Potter?"
        entities = ["books", "Harry Potter"]
        properties = ["of"]
        
        annotator = SimpleSentencePlaceholder()
        self.assertEqual(annotator.annotate(message, entities, properties), "What are the [entity 0] [property 0] [entity 1]?")