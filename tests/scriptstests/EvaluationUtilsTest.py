import sys
from pathlib import Path
sys.path.append(Path("modules").absolute().__str__())

import unittest
from modules.evaluation_utils import (
    process_dataset_for_evaluation,
    is_correct_SPARQL_query,
    unique_metric,
    is_entity_column,
    keep_id_columns,
    transform_serie_into_qrel_list,
    transform_list_into_qrel_list,
    transform_serie_into_run_list,
    transform_list_into_run_list,
)
from modules.execution_utils import add_relevant_prefixes_to_query
import pandas as pd
import ir_measures

path_to_data = Path("tests/scriptstests/tmp/evaluation_handmade.parquet.gzip")

class EvaluationUtilsTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        
    @classmethod
    def setUpClass(cls) -> None:
        path_to_data.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "has_error": [
                False,
                True,
                False,
                False,
                False,
            ],
            "execution":[
                """[
                    {"id": {"type": "literal", "value": "Q0"}, "idLabel": {"type": "literal", "value": "An example label"}}, 
                    {"id": {"type": "literal", "value": "Q1"}, "idLabel": {"type": "literal", "value": "A second example"}}
                ]""",
                "exception: query is empty",
                "exception: something happened",
                "timeout",
                '[]',
            ]
        }
        
        df_data = pd.DataFrame.from_dict(data)
        df_data.to_parquet(str(path_to_data), engine="fastparquet", compression="gzip")
        # df_data.to_json(str(path_to_data))
        
        return super().setUpClass()
    
    @classmethod
    def tearDownClass(cls) -> None:
        path_to_data.unlink()
        
        return super().tearDownClass()
    
    def test_process_dataset_for_evaluation_normal(self):
        df,df_exec_timeout,df_exec_fail,df_exec_empty,df_exec_to_eval,df_eval = process_dataset_for_evaluation(path_to_data)
        
        self.assertEqual(5, len(df))
        
        self.assertEqual(1, len(df_exec_timeout))
        self.assertEqual(1, len(df_exec_fail))
        self.assertEqual(1, len(df_exec_empty))
        self.assertEqual(1, len(df_exec_to_eval))
        self.assertEqual(1, len(df_eval))
        
        self.assertListEqual(['has_error', 'execution'], list(df.columns))
        self.assertListEqual(['has_error', 'execution'], list(df_exec_fail.columns))
        self.assertListEqual(['has_error', 'execution'], list(df_exec_empty.columns))
        self.assertListEqual(['has_error', 'execution'], list(df_exec_to_eval.columns))
        self.assertListEqual(['has_error', 'execution', 'eval', 'get_nested_values', 'eval_df', 'id_columns'], list(df_eval.columns))
        
    def test_is_correct_sparql_query_empty(self):
        query = ""
        self.assertFalse(is_correct_SPARQL_query(query))
        
    def test_is_correct_sparql_query_not_a_query(self):
        query = "not a query"
        self.assertFalse(is_correct_SPARQL_query(query))
        
    def test_is_correct_sparql_query_a_random_paragraph(self):
        query = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Morbi et leo ullamcorper, iaculis mauris at, efficitur mauris. Suspendisse condimentum felis nisi, sed suscipit orci vestibulum sed. Morbi interdum nulla eu vehicula cursus. Maecenas arcu libero, placerat elementum eleifend venenatis, volutpat a nisl. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Curabitur et ex eu tortor tincidunt semper. Nulla consequat lectus vitae elit facilisis rutrum. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer eu ex pretium, eleifend ex at, posuere orci. Curabitur sit amet mauris velit. Ut bibendum, leo sed hendrerit pretium, ligula nibh pharetra arcu, nec sodales eros nisl ut quam. Nulla porttitor, metus id malesuada blandit, orci sapien viverra nibh, in consectetur magna eros ultrices elit. Duis eu purus id nisi cursus accumsan. Sed eleifend in justo ac mollis. Suspendisse potenti. Vivamus bibendum auctor arcu, quis mattis augue tincidunt eget. """
        self.assertFalse(is_correct_SPARQL_query(query))
        
    def test_is_correct_sparql_query_none(self):
        query = None
        self.assertFalse(is_correct_SPARQL_query(query))
        
    def test_is_correct_sparql_query_list(self):
        query = ['why', 'a', 'list', '?']
        self.assertFalse(is_correct_SPARQL_query(query))
        
    def test_is_correct_sparql_query_correct_query(self):
        query = """SELECT ?property ?propertyType ?propertyLabel ?propertyDescription WHERE {\n?property wikibase:propertyType ?propertyType .\nSERVICE wikibase:label { bd:serviceParam wikibase:language \"[AUTO_LANGUAGE],en\". }\n} ORDER BY ASC(xsd:integer(STRAFTER(STR(?property), 'P')))"""
        self.assertTrue(is_correct_SPARQL_query(query))
        
    def test_is_correct_sparql_query_correct_query_with_prefixes(self):
        query = """SELECT ?property ?propertyType ?propertyLabel ?propertyDescription WHERE {\n?property wikibase:propertyType ?propertyType .\nSERVICE wikibase:label { bd:serviceParam wikibase:language \"[AUTO_LANGUAGE],en\". }\n} ORDER BY ASC(xsd:integer(STRAFTER(STR(?property), 'P')))"""
        self.assertTrue(is_correct_SPARQL_query(add_relevant_prefixes_to_query(query)))
        
    def test_unique_metric_all_serie_data_is_unique(self):
        column = pd.Series(data=["1", "2", "3", "4"])
        self.assertEqual(1, unique_metric(column))
        
    def test_unique_metric_all_list_data_is_unique(self):
        column = ["1", "2", "3", "4"]
        self.assertEqual(1, unique_metric(column))
        
    def test_unique_metric_empty_serie_data(self):
        column = pd.Series()
        self.assertEqual(0, unique_metric(column))
        
    def test_unique_metric_empty_list_data(self):
        column = []
        self.assertEqual(0, unique_metric(column))
        
    def test_unique_metric_None_filled_serie_data(self):
        column = pd.Series(data=[None, None, None])
        self.assertAlmostEqual(1/3, unique_metric(column))
        
    def test_unique_metric_None_filled_list_data(self):
        column = [None, None, None]
        self.assertAlmostEqual(1/3, unique_metric(column))
        
    def test_unique_metric_half_serie_data_is_unique(self):
        column = pd.Series(data=["1", "1", "3", "3"])
        self.assertEqual(0.5, unique_metric(column))
        
    def test_unique_metric_all_list_data_is_unique(self):
        column = ["2", "2", "3", "3"]
        self.assertEqual(0.5, unique_metric(column))
    
    def test_is_entity_column_not_str_list(self):
        iterable = [0, 1, 2, 3]
        self.assertFalse(is_entity_column(iterable))
    
    def test_is_entity_column_not_str_series(self):
        iterable = pd.Series(data=[0, 1, 2, 3])
        self.assertFalse(is_entity_column(iterable))
    
    def test_is_entity_column_str_list_not_entities(self):
        iterable = ["0", "1", "2", "3"]
        self.assertFalse(is_entity_column(iterable))
    
    def test_is_entity_column_str_series_not_entities(self):
        iterable = pd.Series(data=["0", "1", "2", "3"])
        self.assertFalse(is_entity_column(iterable))
    
    def test_is_entity_column_str_list_entities(self):
        iterable = ["http://www.wikidata.org/entity/Q0", "http://www.wikidata.org/entity/Q1", "http://www.wikidata.org/entity/Q2", "http://www.wikidata.org/entity/Q3"]
        self.assertTrue(is_entity_column(iterable))
    
    def test_is_entity_column_str_series_entities(self):
        iterable = pd.Series(data=["http://www.wikidata.org/entity/Q0", "http://www.wikidata.org/entity/Q1", "http://www.wikidata.org/entity/Q2", "http://www.wikidata.org/entity/Q3"])
        self.assertTrue(is_entity_column(iterable))
    
    def test_is_entity_column_str_list_part_not_entities(self):
        iterable = ["http://www.wikidata.org/entity/Q0", "Q1", "http://www.wikidata.org/entity/Q2", "Q3"]
        self.assertFalse(is_entity_column(iterable))
    
    def test_is_entity_column_str_series_part_not_entities(self):
        iterable = pd.Series(data=["http://www.wikidata.org/entity/Q0", "Q1", "http://www.wikidata.org/entity/Q2", "Q3"])
        self.assertFalse(is_entity_column(iterable))
    
    def test_is_entity_column_str_list_has_none_elements(self):
        iterable = pd.Series(data=["http://www.wikidata.org/entity/Q0", None, "http://www.wikidata.org/entity/Q2", "http://www.wikidata.org/entity/Q0"])
        self.assertFalse(is_entity_column(iterable))
    
    def test_keep_id_columns_one_column(self):
        data = { "alone" : [ "a", "b", "c" ] }
        
        df_data = pd.DataFrame.from_dict(data)
        
        id_columns = keep_id_columns(df_data)
        self.assertListEqual(["alone"], list(id_columns.columns))
        
    def test_keep_id_columns_two_columns_one_with_unique_values(self):
        data = { "unique" : [ "a", "b", "c" ], "not unique": [ "1", "1", "1" ] }
        
        df_data = pd.DataFrame.from_dict(data)
        
        id_columns = keep_id_columns(df_data)
        self.assertListEqual(["unique"], list(id_columns.columns))
        
    def test_keep_id_columns_two_columns_two_with_unique_values(self):
        data = { "unique" : [ "a", "b", "c" ], "unique too": [ "1", "2", "3" ] }
        
        df_data = pd.DataFrame.from_dict(data)
        
        id_columns = keep_id_columns(df_data)
        self.assertListEqual(["unique", "unique too"], list(id_columns.columns))
        
    def test_keep_id_columns_empty_data(self):
        df_data = pd.DataFrame()
        id_columns = keep_id_columns(df_data)
        self.assertListEqual([], list(id_columns.columns))
        
    def test_keep_id_columns_data_with_entity_column(self):
        data = { "unique" : [ "a", "b", "c" ], "unique too": [ "1", "2", "3" ], "entity" : ["http://www.wikidata.org/entity/Q0", "http://www.wikidata.org/entity/Q1", "http://www.wikidata.org/entity/Q2"] }
        
        df_data = pd.DataFrame(data)
        id_columns = keep_id_columns(df_data)
        self.assertListEqual(["entity"], list(id_columns.columns))
        
    def test_keep_id_columns_two_columns_two_with_unique_values_but_id_in_name(self):
        data = { "unique id" : [ "a", "b", "c" ], "unique too": [ "1", "2", "3" ]}
        
        df_data = pd.DataFrame(data)
        id_columns = keep_id_columns(df_data)
        self.assertListEqual(["unique id"], list(id_columns.columns))
        
    def test_keep_id_columns_two_columns_two_with_unique_values_but_id_in_name_2(self):
        data = { "id unique" : [ "a", "b", "c" ], "unique too": [ "1", "2", "3" ]}
        
        df_data = pd.DataFrame(data)
        id_columns = keep_id_columns(df_data)
        self.assertListEqual(["id unique"], list(id_columns.columns))
    
    def test_transform_serie_into_qrel_list_normal_serie(self):
        serie = pd.Series(data=["a", "b", "c"])
        qid = "Q0"
        
        expected = [
            ir_measures.Qrel(qid, "a", 1),
            ir_measures.Qrel(qid, "b", 1),
            ir_measures.Qrel(qid, "c", 1),
            ]
        
        res = transform_serie_into_qrel_list(qid, serie)
        
        self.assertEqual(3, len(res))
        self.assertListEqual(expected, res)
    
    def test_transform_serie_into_qrel_list_empty_serie(self):
        serie = pd.Series()
        qid = "Q0"
        
        expected = []
        
        res = transform_serie_into_qrel_list(qid, serie)
        
        self.assertEqual(0, len(res))
        self.assertListEqual(expected, res)
    
    def test_transform_serie_into_qrel_list_none_in_serie(self):
        serie = pd.Series(data=["a", None, "c"])
        qid = "Q0"
        
        expected = [
            ir_measures.Qrel(qid, "a", 1),
            ir_measures.Qrel(qid, "c", 1),
            ]
        
        res = transform_serie_into_qrel_list(qid, serie)
        
        self.assertEqual(2, len(res))
        self.assertListEqual(expected, res)
    
    def test_transform_list_into_qrel_list_normal_a_list(self):
        a_list = ["a", "b", "c"]
        qid = "Q0"
        
        expected = [
            ir_measures.Qrel(qid, "a", 1),
            ir_measures.Qrel(qid, "b", 1),
            ir_measures.Qrel(qid, "c", 1),
            ]
        
        res = transform_list_into_qrel_list(qid, a_list)
        
        self.assertEqual(3, len(res))
        self.assertListEqual(expected, res)
    
    def test_transform_list_into_qrel_list_empty_a_list(self):
        a_list = []
        qid = "Q0"
        
        expected = []
        
        res = transform_list_into_qrel_list(qid, a_list)
        
        self.assertEqual(0, len(res))
        self.assertListEqual(expected, res)
    
    def test_transform_list_into_qrel_list_none_in_a_list(self):
        a_list = ["a", None, "c"]
        qid = "Q0"
        
        expected = [
            ir_measures.Qrel(qid, "a", 1),
            ir_measures.Qrel(qid, "c", 1),
            ]
        
        res = transform_list_into_qrel_list(qid, a_list)
        
        self.assertEqual(2, len(res))
        self.assertListEqual(expected, res)
        
    def test_transform_serie_into_run_list_normal_serie(self):
        serie = pd.Series(data=["a", "b", "c"])
        qid = "Q0"
        
        expected = [
            ir_measures.ScoredDoc(qid, "a", 1),
            ir_measures.ScoredDoc(qid, "b", 1),
            ir_measures.ScoredDoc(qid, "c", 1),
            ]
        
        res = transform_serie_into_run_list(qid, serie)
        
        self.assertEqual(3, len(res))
        self.assertListEqual(expected, res)
    
    def test_transform_serie_into_run_list_empty_serie(self):
        serie = pd.Series()
        qid = "Q0"
        
        expected = []
        
        res = transform_serie_into_run_list(qid, serie)
        
        self.assertEqual(0, len(res))
        self.assertListEqual(expected, res)
    
    def test_transform_serie_into_run_list_none_in_serie(self):
        serie = pd.Series(data=["a", None, "c"])
        qid = "Q0"
        
        expected = [
            ir_measures.ScoredDoc(qid, "a", 1),
            ir_measures.ScoredDoc(qid, "c", 1),
            ]
        
        res = transform_serie_into_run_list(qid, serie)
        
        self.assertEqual(2, len(res))
        self.assertListEqual(expected, res)
    
    def test_transform_list_into_run_list_normal_a_list(self):
        a_list = ["a", "b", "c"]
        qid = "Q0"
        
        expected = [
            ir_measures.ScoredDoc(qid, "a", 1),
            ir_measures.ScoredDoc(qid, "b", 1),
            ir_measures.ScoredDoc(qid, "c", 1),
            ]
        
        res = transform_list_into_run_list(qid, a_list)
        
        self.assertEqual(3, len(res))
        self.assertListEqual(expected, res)
    
    def test_transform_list_into_run_list_empty_a_list(self):
        a_list = []
        qid = "Q0"
        
        expected = []
        
        res = transform_list_into_run_list(qid, a_list)
        
        self.assertEqual(0, len(res))
        self.assertListEqual(expected, res)
    
    def test_transform_list_into_run_list_none_in_a_list(self):
        a_list = ["a", None, "c"]
        qid = "Q0"
        
        expected = [
            ir_measures.ScoredDoc(qid, "a", 1),
            ir_measures.ScoredDoc(qid, "c", 1),
            ]
        
        res = transform_list_into_run_list(qid, a_list)
        
        self.assertEqual(2, len(res))
        self.assertListEqual(expected, res)