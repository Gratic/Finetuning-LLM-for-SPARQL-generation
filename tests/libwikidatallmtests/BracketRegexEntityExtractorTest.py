import unittest
from modules.libwikidatallm.EntityExtractor import BracketRegexEntityExtractor

class BracketRegexEntityExtractorTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        
    def test_extract_entities_and_properties_normal(self):
        text = """C'est une phrase normal [property:sans être différente] d'une toute autre [entity:phrase]. Pourtant des [entity:accolades] viennent de [property:nulle part]."""
        test_results = (['phrase', 'accolades'], ['sans être différente', 'nulle part'])
        
        bre = BracketRegexEntityExtractor()
        
        result = bre.extract_entities_and_properties(text)
        
        self.assertIsInstance(result, tuple)
        self.assertEqual(2, len(result))
        self.assertIsInstance(result[0], list)
        self.assertIsInstance(result[1], list)
        
        self.assertListEqual(test_results[0], result[0])
        self.assertListEqual(test_results[1], result[1])
        
        

    def test_extract_entities_and_properties_query_normal(self):
        query = """SELECT ?id ?idLabel ?idDescription ?new{
?id wikibase:[property:directClaim] ?pid .
minus{?id wikibase:[property:propertyType] wikibase:[entity:ExternalId]}
BIND(Replace(STR(?id),"http://www.wikidata.org/entity/P"," ") as ?new)
SERVICE wikibase:[entity:label] { bd:serviceParam wikibase:[entity:language] "[AUTO_LANGUAGE],en" }
}
ORDER BY DESC(xsd:integer(?new))"""

        test_results = (['ExternalId', 'label', 'language'], ['directClaim','propertyType'])
        
        bre = BracketRegexEntityExtractor()
        
        result = bre.extract_entities_and_properties(query)
        
        self.assertIsInstance(result, tuple)
        self.assertEqual(2, len(result))
        self.assertIsInstance(result[0], list)
        self.assertIsInstance(result[1], list)
        
        self.assertListEqual(test_results[0], result[0])
        self.assertListEqual(test_results[1], result[1])
    
    def test_execute_queries_base_columns(self):
        text = """C'est une phrase normal [property:sans être différente] d'une toute autre [entity:phrase]. Pourtant des [entity:accolades] viennent de [property:nulle part]."""
        test_results = (['phrase', 'accolades'], ['sans être différente', 'nulle part'])
        
        bre = BracketRegexEntityExtractor()
        
        context = {'row': text}
        
        bre.execute(context)
        
        self.assertTrue('row' in context.keys())
        self.assertTrue('extracted_entities' in context.keys())
        self.assertTrue('extracted_properties' in context.keys())
        
        self.assertListEqual(test_results[0], context['extracted_entities'])
        self.assertListEqual(test_results[1], context['extracted_properties'])
        
    def test_execute_queries_modified_columns(self):
        text = """C'est une phrase normal [property:sans être différente] d'une toute autre [entity:phrase]. Pourtant des [entity:accolades] viennent de [property:nulle part]."""
        test_results = (['phrase', 'accolades'], ['sans être différente', 'nulle part'])
        
        bre = BracketRegexEntityExtractor(
            input_column='translated_prompt',
            output_col_entities='the_entities',
            output_col_properties='the_properties'
        )
        
        context = {'translated_prompt': text, 'row': "[entity:other text], [property: that doesn't matter]"}
        
        bre.execute(context)
        
        self.assertTrue('translated_prompt' in context.keys())
        self.assertTrue('the_entities' in context.keys())
        self.assertTrue('the_properties' in context.keys())
        
        self.assertListEqual(test_results[0], context['the_entities'])
        self.assertListEqual(test_results[1], context['the_properties'])