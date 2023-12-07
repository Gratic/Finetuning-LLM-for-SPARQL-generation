import unittest
from unittest.mock import patch, MagicMock
from libwikidatallm.EntityFinder import WikidataAPI

class WikidataAPITest(unittest.TestCase):
    @patch('libwikidatallm.EntityFinder.requests')
    def test_find_entities_empty_response(self, mock_request):
        # mocking
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"search": []}
        
        mock_request.get.return_value = mock_response
        
        wikidata = WikidataAPI()
        with self.assertRaises(ValueError):
            wikidata.find_entities("test")
            
    @patch('libwikidatallm.EntityFinder.requests')
    def test_find_properties_empty_response(self, mock_request):
        # mocking
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"search": []}
        
        mock_request.get.return_value = mock_response
        
        wikidata = WikidataAPI()
        with self.assertRaises(ValueError):
            wikidata.find_properties("test")
        