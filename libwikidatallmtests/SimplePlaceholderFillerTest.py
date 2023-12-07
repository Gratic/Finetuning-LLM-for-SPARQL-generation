import unittest
from libwikidatallm import SimplePlaceholderFiller

class SimplePlaceholderFillerTest(unittest.TestCase):
    def test_deannotate_annotated_sentence(self):
        annotated_sentence = "This is an [property 0] [entity 0]."
        linked_properties = [("instance of", ("P12", "instance"))]
        linked_entities = [("book", ("Q40", "Book"))]
        
        spf = SimplePlaceholderFiller()
        self.assertEqual(spf.deannotate(annotated_sentence, linked_entities, linked_properties), "This is an P12 Q40.")
        
    def test_deannotate_annotated_sentence_2(self):
        annotated_sentence = "This is an [property 0] [entity 1]."
        linked_properties = [("instance of", ("P12", "instance of"))]
        linked_entities = [("book", ("Q40", "Book")), ("film", ("Q53", "Film"))]
        
        spf = SimplePlaceholderFiller()
        self.assertEqual(spf.deannotate(annotated_sentence, linked_entities, linked_properties), "This is an P12 Q53.")