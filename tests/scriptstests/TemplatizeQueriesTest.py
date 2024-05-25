import sys
from pathlib import Path
sys.path.append(Path("modules").absolute().__str__())

import unittest
from scripts.templatize_queries import replace_entities_and_properties_id_with_labels, extract_entities_properties_ids 

class TemplatizeQueriesTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
    
    def test_extract_entities_properties_ids_extracts_entity_id(self):
        query = "SELECT ?x WHERE {?y :Q100 a :Person}."  # Assuming these are valid Wikidata IDs
        expected = ['Q100']
        self.assertEqual(extract_entities_properties_ids(query), expected)

    def test_extract_entities_properties_ids_extracts_property_id(self):
        query = "SELECT ?x WHERE {?y :P2345 ?z}."  # Assuming these are valid Wikidata IDs
        expected = ['P2345']
        self.assertEqual(extract_entities_properties_ids(query), expected)

    def test_extract_entities_properties_ids_extracts_lexeme_id(self):
        query = "SELECT ?x WHERE {?y :L12345 ?z}."  # Assuming these are valid Wikidata IDs
        expected = ['L12345']
        self.assertEqual(extract_entities_properties_ids(query), expected)

    def test_extract_entities_properties_ids_returns_empty_list_when_no_match(self):
        query = "SELECT ?x WHERE {?y :X999 a :Person}."  # Assuming this is not a valid Wikidata ID
        self.assertEqual(extract_entities_properties_ids(query), [])

    def test_extract_entities_properties_ids_returns_multiple_matches(self):
        query = "SELECT ?x WHERE {?y :Q100 a :Person; :P2345 ?z}."  # Assuming these are valid Wikidata IDs
        expected = ['Q100', 'P2345']
        self.assertEqual(extract_entities_properties_ids(query), expected)
        
    def test_extract_entities_properties_ids_fq17_1638(self):
        query = "SELECT\n?person # find persons ...\n?personLabel # ... their label\n?secondsInSpace # ... seconds spent in space\n(GROUP_CONCAT(DISTINCT ?countryLabel; # ... and all operating countries\nSEPARATOR=\", \") AS ?countries)\nWITH {\n# this subquery finds all humans on spaceflights and their operators\nSELECT ?person ?flight ?country WHERE {\n?flight wdt:P31 wd:Q752783 ; # human spaceflight ...\nwdt:P620 [] ; # ... has landed\nwdt:P619 [] ; # ... has launched\nwdt:P137\/wdt:P17 ?country ; # ... sovereign state of operator\nwdt:P1029 ?person . # ... with ?person as crew member\n}\n} AS %spaceflight WITH {\n# this subquery finds two distinct launch operators for humans on\n# spaceflights. we order countries A and B by their URI, so that we\n# can more easily filter out duplicates\nSELECT DISTINCT ?person ?countryA ?countryB WHERE {\n{ SELECT ?person ?flightA ?countryA WHERE { # find a spaceflight for ?person\nINCLUDE %spaceflight .\nBIND(?flight AS ?flightA)\nBIND(?country AS ?countryA)\n}}\n{ SELECT ?person ?flightB ?countryB WHERE { # find a second spaceflight for the ?person\nINCLUDE %spaceflight .\nBIND(?flight AS ?flightB)\nBIND(?country AS ?countryB)\n}}\nFILTER (STR(?flightA) < STR(?flightB)) # enforce that the launches are distinct\nFILTER (STR(?countryA) < STR(?countryB)) # enforce that the operating countries are distinct\n}} AS %flightCountries\nWHERE {\n{ SELECT ?person ?country WHERE { # now get both results into the same column\n{ SELECT ?person ?country WHERE { # find ?countryA for ?person\nINCLUDE %flightCountries .\nBIND(?countryA AS ?country)\n}} UNION { SELECT ?person ?country WHERE { # find ?countryB for ?person\nINCLUDE %flightCountries .\nBIND(?countryB AS ?country)\n}}\n}}\n?person wdt:P2873 ?secondsInSpace . # also get the time spent in space\nSERVICE wikibase:label { # finally, get the labels ...\nbd:serviceParam wikibase:language \"[AUTO_LANGUAGE],en\" .\n?person rdfs:label ?personLabel . # ... for person\n?country rdfs:label ?countryLabel . # ... and for the country\n}\n} GROUP BY ?person ?secondsInSpace ?personLabel # group by ?person ...\nORDER BY DESC(?secondsInSpace) ASC(?personLabel) # ... and order by time in space and name"
        expected = ['P31', "Q752783", "P620", "P619", "P137", "P17", "P1029", "P2873"]
        self.assertListEqual(extract_entities_properties_ids(query), expected)
        
    def test_replace_entities_and_properties_id_with_labels_fq17_1638(self):
        query = "SELECT\n?person # find persons ...\n?personLabel # ... their label\n?secondsInSpace # ... seconds spent in space\n(GROUP_CONCAT(DISTINCT ?countryLabel; # ... and all operating countries\nSEPARATOR=\", \") AS ?countries)\nWITH {\n# this subquery finds all humans on spaceflights and their operators\nSELECT ?person ?flight ?country WHERE {\n?flight wdt:P31 wd:Q752783 ; # human spaceflight ...\nwdt:P620 [] ; # ... has landed\nwdt:P619 [] ; # ... has launched\nwdt:P137\/wdt:P17 ?country ; # ... sovereign state of operator\nwdt:P1029 ?person . # ... with ?person as crew member\n}\n} AS %spaceflight WITH {\n# this subquery finds two distinct launch operators for humans on\n# spaceflights. we order countries A and B by their URI, so that we\n# can more easily filter out duplicates\nSELECT DISTINCT ?person ?countryA ?countryB WHERE {\n{ SELECT ?person ?flightA ?countryA WHERE { # find a spaceflight for ?person\nINCLUDE %spaceflight .\nBIND(?flight AS ?flightA)\nBIND(?country AS ?countryA)\n}}\n{ SELECT ?person ?flightB ?countryB WHERE { # find a second spaceflight for the ?person\nINCLUDE %spaceflight .\nBIND(?flight AS ?flightB)\nBIND(?country AS ?countryB)\n}}\nFILTER (STR(?flightA) < STR(?flightB)) # enforce that the launches are distinct\nFILTER (STR(?countryA) < STR(?countryB)) # enforce that the operating countries are distinct\n}} AS %flightCountries\nWHERE {\n{ SELECT ?person ?country WHERE { # now get both results into the same column\n{ SELECT ?person ?country WHERE { # find ?countryA for ?person\nINCLUDE %flightCountries .\nBIND(?countryA AS ?country)\n}} UNION { SELECT ?person ?country WHERE { # find ?countryB for ?person\nINCLUDE %flightCountries .\nBIND(?countryB AS ?country)\n}}\n}}\n?person wdt:P2873 ?secondsInSpace . # also get the time spent in space\nSERVICE wikibase:label { # finally, get the labels ...\nbd:serviceParam wikibase:language \"[AUTO_LANGUAGE],en\" .\n?person rdfs:label ?personLabel . # ... for person\n?country rdfs:label ?countryLabel . # ... and for the country\n}\n} GROUP BY ?person ?secondsInSpace ?personLabel # group by ?person ...\nORDER BY DESC(?secondsInSpace) ASC(?personLabel) # ... and order by time in space and name"
        expected = "SELECT\n?person # find persons ...\n?personLabel # ... their label\n?secondsInSpace # ... seconds spent in space\n(GROUP_CONCAT(DISTINCT ?countryLabel; # ... and all operating countries\nSEPARATOR=\", \") AS ?countries)\nWITH {\n# this subquery finds all humans on spaceflights and their operators\nSELECT ?person ?flight ?country WHERE {\n?flight wdt:[property:instance of] wd:[entity:human spaceflight] ; # human spaceflight ...\nwdt:[property:UTC date of spacecraft landing] [] ; # ... has landed\nwdt:[property:UTC date of spacecraft launch] [] ; # ... has launched\nwdt:[property:operator]\/wdt:[property:country] ?country ; # ... sovereign state of operator\nwdt:[property:crew member(s)] ?person . # ... with ?person as crew member\n}\n} AS %spaceflight WITH {\n# this subquery finds two distinct launch operators for humans on\n# spaceflights. we order countries A and B by their URI, so that we\n# can more easily filter out duplicates\nSELECT DISTINCT ?person ?countryA ?countryB WHERE {\n{ SELECT ?person ?flightA ?countryA WHERE { # find a spaceflight for ?person\nINCLUDE %spaceflight .\nBIND(?flight AS ?flightA)\nBIND(?country AS ?countryA)\n}}\n{ SELECT ?person ?flightB ?countryB WHERE { # find a second spaceflight for the ?person\nINCLUDE %spaceflight .\nBIND(?flight AS ?flightB)\nBIND(?country AS ?countryB)\n}}\nFILTER (STR(?flightA) < STR(?flightB)) # enforce that the launches are distinct\nFILTER (STR(?countryA) < STR(?countryB)) # enforce that the operating countries are distinct\n}} AS %flightCountries\nWHERE {\n{ SELECT ?person ?country WHERE { # now get both results into the same column\n{ SELECT ?person ?country WHERE { # find ?countryA for ?person\nINCLUDE %flightCountries .\nBIND(?countryA AS ?country)\n}} UNION { SELECT ?person ?country WHERE { # find ?countryB for ?person\nINCLUDE %flightCountries .\nBIND(?countryB AS ?country)\n}}\n}}\n?person wdt:[property:time in space] ?secondsInSpace . # also get the time spent in space\nSERVICE wikibase:label { # finally, get the labels ...\nbd:serviceParam wikibase:language \"[AUTO_LANGUAGE],en\" .\n?person rdfs:label ?personLabel . # ... for person\n?country rdfs:label ?countryLabel . # ... and for the country\n}\n} GROUP BY ?person ?secondsInSpace ?personLabel # group by ?person ...\nORDER BY DESC(?secondsInSpace) ASC(?personLabel) # ... and order by time in space and name"

        result = replace_entities_and_properties_id_with_labels(query)
        
        self.assertEqual(result, expected)
    
    def test_extract_entities_properties_ids_fq17_1639(self):
        query = "select distinct ?item ?itemLabel ?leadership_start_time (MAX(?elec_date) AS ?elec_date) (MIN(?days) AS ?days)\nwith {\nselect ?elec ?elec_date WHERE {\n?elec wdt:P31 wd:Q15283424 .\n?elec wdt:P585 ?elec_date .\n}\n} AS %elections\nwhere {\nhint:Query hint:optimizer \"None\".\n?role wdt:P279? wd:Q1553195 .\n?pos_stmt ps:P39 ?role .\n?pos_stmt pq:P580 ?leadership_start_time .\nFILTER (year(?leadership_start_time) > 1945) .\nMINUS {?role p:P279\/pq:P642\/wdt:P31 wd:Q848197} .\n?item p:P39 ?pos_stmt .\n?item wdt:P39\/wdt:P279 wd:Q16707842 .\nINCLUDE %elections .\nFILTER (?elec_date < ?leadership_start_time).\nBIND ((?leadership_start_time - ?elec_date) AS ?days)\nSERVICE wikibase:label { bd:serviceParam wikibase:language \"[AUTO_LANGUAGE],en\". } .\n} GROUP BY ?item ?itemLabel ?leadership_start_time\nORDER BY DESC(?days)"
        expected = ['P31', "Q15283424", "P585", "P279", "Q1553195", "P39", "P580", "P279", "P642", "P31", "Q848197", "P39", "P39", "P279", "Q16707842"]
        self.assertListEqual(extract_entities_properties_ids(query), expected)
    
    def test_replace_entities_and_properties_id_with_labels_fq17_1639(self):
        query = "select distinct ?item ?itemLabel ?leadership_start_time (MAX(?elec_date) AS ?elec_date) (MIN(?days) AS ?days)\nwith {\nselect ?elec ?elec_date WHERE {\n?elec wdt:P31 wd:Q15283424 .\n?elec wdt:P585 ?elec_date .\n}\n} AS %elections\nwhere {\nhint:Query hint:optimizer \"None\".\n?role wdt:P279? wd:Q1553195 .\n?pos_stmt ps:P39 ?role .\n?pos_stmt pq:P580 ?leadership_start_time .\nFILTER (year(?leadership_start_time) > 1945) .\nMINUS {?role p:P279\/pq:P642\/wdt:P31 wd:Q848197} .\n?item p:P39 ?pos_stmt .\n?item wdt:P39\/wdt:P279 wd:Q16707842 .\nINCLUDE %elections .\nFILTER (?elec_date < ?leadership_start_time).\nBIND ((?leadership_start_time - ?elec_date) AS ?days)\nSERVICE wikibase:label { bd:serviceParam wikibase:language \"[AUTO_LANGUAGE],en\". } .\n} GROUP BY ?item ?itemLabel ?leadership_start_time\nORDER BY DESC(?days)"
        expected = "select distinct ?item ?itemLabel ?leadership_start_time (MAX(?elec_date) AS ?elec_date) (MIN(?days) AS ?days)\nwith {\nselect ?elec ?elec_date WHERE {\n?elec wdt:[property:instance of] wd:[entity:United Kingdom general election] .\n?elec wdt:[property:point in time] ?elec_date .\n}\n} AS %elections\nwhere {\nhint:Query hint:optimizer \"None\".\n?role wdt:[property:subclass of]? wd:[entity:party leader] .\n?pos_stmt ps:[property:position held] ?role .\n?pos_stmt pq:[property:start time] ?leadership_start_time .\nFILTER (year(?leadership_start_time) > 1945) .\nMINUS {?role p:[property:subclass of]\/pq:[property:of]\/wdt:[property:instance of] wd:[entity:parliamentary group]} .\n?item p:[property:position held] ?pos_stmt .\n?item wdt:[property:position held]\/wdt:[property:subclass of] wd:[entity:Member of Parliament] .\nINCLUDE %elections .\nFILTER (?elec_date < ?leadership_start_time).\nBIND ((?leadership_start_time - ?elec_date) AS ?days)\nSERVICE wikibase:label { bd:serviceParam wikibase:language \"[AUTO_LANGUAGE],en\". } .\n} GROUP BY ?item ?itemLabel ?leadership_start_time\nORDER BY DESC(?days)"

        result = replace_entities_and_properties_id_with_labels(query)
        
        self.assertEqual(result, expected)
    
    def test_replace_entities_and_properties_id_with_labels_fq17_2652(self):
        query = """SELECT (count(?album) as ?albums) ?P407Label ?P407 WHERE {
SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
?album wdt:P31 wd:Q482994.
OPTIONAL { ?album wdt:P407 ?P407. }
}
GROUP BY ?P407 ?P407Label
ORDER BY DESC(?albums) ASC(?P407)
#Do you know how to improve this query; make it smarter, better, more elegant?
#If you do, please don't hesitate to drop us a line at User_talk:Moebeus or join
#the conversation over on Telegram: https://t.me/exmusica"""
        expected = """SELECT (count(?album) as ?albums) ?P407Label ?P407 WHERE {
SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
?album wdt:[property:instance of] wd:[entity:album].
OPTIONAL { ?album wdt:[property:language of work or name] ?P407. }
}
GROUP BY ?P407 ?P407Label
ORDER BY DESC(?albums) ASC(?P407)
#Do you know how to improve this query; make it smarter, better, more elegant?
#If you do, please don't hesitate to drop us a line at User_talk:Moebeus or join
#the conversation over on Telegram: https://t.me/exmusica"""

        result = replace_entities_and_properties_id_with_labels(query)
        
        self.assertEqual(result, expected)