from typing import Any
import unittest
from execute_queries import can_add_limit_clause, is_query_empty, send_query_to_api, add_relevant_prefixes_to_query
from libwikidatallm.EntityFinder import WikidataAPI
import requests

class QueryGenerator():
    def __init__(self) -> None:
        pass
    
    def __call__(self, id, *args: Any, **kwds: Any) -> Any:
        if id == 0:
            return "A normal query"
        elif id == 1:
            return ""
        elif id == 2:
            return " "
        elif id == 3:
            return "                       "
        elif id == 4:
            return "A query containing a LIMIT clause"
        elif id == 5:
            return "A query COUNT -ing something"
        elif id == 6:
            return "A query containing a LIMIT clause and a COUNT -ing clause."
        elif id == 7:
            return "A query containing a limit clause and a COUNT -ing clause."
        elif id == 8:
            return "A query containing a lImIt clause and a COUNT -ing clause."
        elif id == 9:
            return "A query containing a LIMIT clause and a count -ing clause."
        elif id == 10:
            return "A query containing a LIMIT clause and a cOuNt -ing clause."
        elif id == 11:
            return "A query containing a limit clause and a count -ing clause."
        elif id == 12:
            return "A query containing a lImiT clause and a coUnT -ing clause."
        elif id == 13:
            return "A query containing a limit clause"
        elif id == 14:
            return "A query containing a lImiT clause"
        elif id == 15:
            return "A query count -ing something"
        elif id == 16:
            return "A query cOunT -ing something"
        raise ValueError("This id is not supported.")

class MockResponse():
    def __init__(self, success, msg) -> None:
        self.success = success
        self.msg = msg
    
    @property
    def bindings(self):
        return None if not self.success else self.msg
    
    @property
    def data(self):
        return None if self.success else self.msg

class MockAPI(WikidataAPI):
    def __init__(self, base_url: str = "https://www.wikidata.org/w/api.php") -> None:
        super().__init__(base_url)
    
    def execute_sparql(self, query: str, timeout: int = None):
        if query == "valid":
            return MockResponse(True, "success")
        elif query == "retry-after":
            response = requests.Response()
            response.headers['retry-after'] = '1'
            response.status_code = 429
            raise requests.exceptions.HTTPError(request=None, response=response)
        elif query == "400Error":
            response = requests.Response()
            response.status_code = 400
            raise requests.exceptions.HTTPError(response=response)
        elif query == "timeout":
            raise requests.exceptions.Timeout()
            
        return ValueError("This query is not supported.")

class ExecuteQueriesTest(unittest.TestCase):
    
    def setUp(self) -> None:
        self.query = QueryGenerator()
        self.api = MockAPI()
    
    def test_is_query_empty_normal(self):
        self.assertFalse(is_query_empty(self.query(0)))
    
    def test_is_query_empty_empty_0(self):
        self.assertTrue(is_query_empty(self.query(1)))
    
    def test_is_query_empty_empty_1(self):
        self.assertTrue(is_query_empty(self.query(2)))
    
    def test_is_query_empty_empty_2(self):
        self.assertTrue(is_query_empty(self.query(3)))
    
    def test_can_add_limit_clause_valid(self):
        self.assertTrue(can_add_limit_clause(self.query(0)))
        
    def test_can_add_limit_clause_empty_0(self):
        self.assertFalse(can_add_limit_clause(self.query(1)))
        
    def test_can_add_limit_clause_empty_1(self):
        self.assertFalse(can_add_limit_clause(self.query(2)))
        
    def test_can_add_limit_clause_empty_2(self):
        self.assertFalse(can_add_limit_clause(self.query(3)))
        
    def test_can_add_limit_clause_contains_LIMIT(self):
        self.assertFalse(can_add_limit_clause(self.query(4)))
        
    def test_can_add_limit_clause_contains_COUNT(self):
        self.assertFalse(can_add_limit_clause(self.query(5)))
        
    def test_can_add_limit_clause_contains_COUNT_and_LIMIT(self):
        self.assertFalse(can_add_limit_clause(self.query(6)))
        
    def test_can_add_limit_clause_contains_limit_and_COUNT(self):
        self.assertFalse(can_add_limit_clause(self.query(7)))
        
    def test_can_add_limit_clause_contains_lImIt_and_COUNT(self):
        self.assertFalse(can_add_limit_clause(self.query(8)))
        
    def test_can_add_limit_clause_contains_count_and_LIMIT(self):
        self.assertFalse(can_add_limit_clause(self.query(9)))
        
    def test_can_add_limit_clause_contains_cOuNt_and_LIMIT(self):
        self.assertFalse(can_add_limit_clause(self.query(10)))
        
    def test_can_add_limit_clause_contains_count_and_limit(self):
        self.assertFalse(can_add_limit_clause(self.query(11)))
        
    def test_can_add_limit_clause_contains_cOuNt_and_lImIt(self):
        self.assertFalse(can_add_limit_clause(self.query(12)))
        
    def test_can_add_limit_clause_contains_limit(self):
        self.assertFalse(can_add_limit_clause(self.query(13)))
        
    def test_can_add_limit_clause_contains_lImIt(self):
        self.assertFalse(can_add_limit_clause(self.query(14)))
        
    def test_can_add_limit_clause_contains_count(self):
        self.assertFalse(can_add_limit_clause(self.query(15)))
        
    def test_can_add_limit_clause_contains_cOunT(self):
        self.assertFalse(can_add_limit_clause(self.query(16)))
        
    def test_send_query_to_api_valid(self):
        self.assertEqual("success", send_query_to_api("valid", self.api, None, 3))
        
    def test_send_query_to_api_retry_after(self):
        self.assertEqual("exception: too many retry-after", send_query_to_api("retry-after", self.api, None, 3))
    
    def test_send_query_to_api_HTTPError(self):
        self.assertEqual("exception: \n", send_query_to_api("400Error", self.api, None, 3))
        
    def test_send_query_to_api_timeout(self):
        self.assertEqual("timeout", send_query_to_api("timeout", self.api, None, 3))
    
    def test_add_relevant_prefixes_to_query_empty_query(self):
        query = ""
        
        self.assertEqual(query, add_relevant_prefixes_to_query(query))
    
    def test_add_relevant_prefixes_to_query_no_prefix(self):
        query = "empty"
        
        self.assertEqual(query, add_relevant_prefixes_to_query(query))
    
    def test_add_relevant_prefixes_to_query_two_prefixes(self):
        query = """SELECT DISTINCT * WHERE {
?item wdt:P31 wd:Q1032372;
wdt:P625 ?geo .
}"""
            
        result = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT DISTINCT * WHERE {
?item wdt:P31 wd:Q1032372;
wdt:P625 ?geo .
}"""
        
        self.assertEqual(result, add_relevant_prefixes_to_query(query))
    
    def test_add_relevant_prefixes_to_query_prefix_already_there(self):
        query = """PREFIX dct: <http://purl.org/dc/terms/> # workaround for T233148
SELECT ?lexeme ?lemma (STRLEN(?lemma) AS ?length) WHERE {
BIND(wd:Q1860 AS ?language)
?lexeme dct:language ?language;
wikibase:lemma ?lemma.
FILTER(STRLEN(?lemma) >= 10) # cheap filter to avoid running REGEX() on too many lemmata
FILTER(!REGEX(LCASE(?lemma), "(.).*\\1"))
FILTER(!CONTAINS(?lemma, " "))
}
ORDER BY DESC(?length)"""
        
        result = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wikibase: <http://wikiba.se/ontology#>

PREFIX dct: <http://purl.org/dc/terms/> # workaround for T233148
SELECT ?lexeme ?lemma (STRLEN(?lemma) AS ?length) WHERE {
BIND(wd:Q1860 AS ?language)
?lexeme dct:language ?language;
wikibase:lemma ?lemma.
FILTER(STRLEN(?lemma) >= 10) # cheap filter to avoid running REGEX() on too many lemmata
FILTER(!REGEX(LCASE(?lemma), "(.).*\\1"))
FILTER(!CONTAINS(?lemma, " "))
}
ORDER BY DESC(?length)"""
    
        self.assertEqual(result, add_relevant_prefixes_to_query(query))
        
    def test_add_relevant_prefixes_to_query_prefix_already_there_2(self):
        query = """PREFIX dct: <http://purl.org/dc/terms/> # workaround for T233148
SELECT ?lexeme ?lemma (STRLEN(?lemma) AS ?length) WHERE {
BIND(wd:Q1860 AS ?language)
?lexeme dct:language ?language;
wikibase:lemma ?lemma.
FILTER(STRLEN(?lemma) >= 5)
# first version allows repeated letters, second one doesn't
# FILTER(REGEX(?lemma, "^A*B*C*D*E*F*G*H*I*J*K*L*M*N*O*P*Q*R*S*T*U*V*Q*X*Y*Z*$", "i"))
FILTER(REGEX(?lemma, "^A?B?C?D?E?F?G?H?I?J?K?L?M?N?O?P?Q?R?S?T?U?V?Q?X?Y?Z?$", "i"))
}
ORDER BY DESC(?length)"""
        
        result = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wikibase: <http://wikiba.se/ontology#>

PREFIX dct: <http://purl.org/dc/terms/> # workaround for T233148
SELECT ?lexeme ?lemma (STRLEN(?lemma) AS ?length) WHERE {
BIND(wd:Q1860 AS ?language)
?lexeme dct:language ?language;
wikibase:lemma ?lemma.
FILTER(STRLEN(?lemma) >= 5)
# first version allows repeated letters, second one doesn't
# FILTER(REGEX(?lemma, "^A*B*C*D*E*F*G*H*I*J*K*L*M*N*O*P*Q*R*S*T*U*V*Q*X*Y*Z*$", "i"))
FILTER(REGEX(?lemma, "^A?B?C?D?E?F?G?H?I?J?K?L?M?N?O?P?Q?R?S?T?U?V?Q?X?Y?Z?$", "i"))
}
ORDER BY DESC(?length)"""
    
        self.assertEqual(result, add_relevant_prefixes_to_query(query))