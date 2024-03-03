import unittest
from modules.libwikidatallm.EntityFinder import WikidataAPI

class WikidataAPITest(unittest.TestCase):
    def setUp(self) -> None:
        self.api = WikidataAPI()
    
    def test_get_label_correct_label(self):
        item = {
                "id": "test",
                "display": {
                    "label": {
                        "value": "test_label"
                    }
                }
            }
        
        self.assertEqual(('test', "test_label"), self.api._get_label_from_wbsearchentities(item))
    
    def test_get_label_correct_description(self):
        item = {
                "id": "test",
                "display": {
                    "description": {
                        "value": "test_description"
                    }
                }
            }
        
        self.assertEqual(('test', "test_description"), self.api._get_label_from_wbsearchentities(item))
        
    def test_recover_redirected_id_working_example(self):
        input_id = "Q5227308"
        gold_redirected_id = "Q5227240"
        
        redirected_id = self.api._recover_redirected_id(name=input_id, is_property=False)
        self.assertEqual(gold_redirected_id, redirected_id)
        
    def test_recover_redirected_id_no_redirection(self):
        input_id = "Q5227240"
        gold_redirected_id = "Q5227240"
        
        redirected_id = self.api._recover_redirected_id(name=input_id, is_property=False)
        self.assertEqual(gold_redirected_id, redirected_id)