from abc import ABC, abstractmethod
import http.client
import json
import os
from typing import List

POST_COMPLETION_HEADERS = {"Content-Type":"application/json"}

class BaseProvider(ABC):
    def __init__(self) -> None:
        self.last_answer: str = None
        self.last_full_answer: str | dict[str, str] = None

    @abstractmethod
    def query(self, parameters: dict[str, str | int | float]) -> bool:
        pass

    def get_full_answer(self):
        '''Full result from the provider.'''
        return self.last_full_answer

    def get_answer(self):
        '''Only the generated answer.'''
        return self.last_answer

    @abstractmethod
    def get_tokens(self, parameters: dict[str, str | int | float]) -> List[int]:
        pass

class ServerProvider(BaseProvider):
    def __init__(self, server_address: str, server_port: str, completion_endpoint: str, tokenizer_endpoint: str, post_completion_headers: str = POST_COMPLETION_HEADERS, post_tokenizer_headers: str = POST_COMPLETION_HEADERS) -> None:
        super().__init__()
        self.server_addr = server_address
        self.server_port = server_port
        self.server_completion_endpoint = completion_endpoint
        self.server_tokenizer_endpoint = tokenizer_endpoint
        self.post_completion_headers = post_completion_headers
        self.post_tokenizer_headers = post_tokenizer_headers
    
    def query(self, parameters: dict[str, str | int | float]) -> bool:
        body_json = json.dumps(parameters)
        connection = http.client.HTTPConnection(f"{self.server_addr}:{self.server_port}")
        connection.request(method="POST",
                url=self.server_completion_endpoint,
                headers=self.post_completion_headers, 
                body=body_json)

        response = connection.getresponse()
        self.last_response = response

        if response.status != 200:
            return False
        
        answer = response.read()
        answer_dict = json.loads(answer)

        self.last_answer = answer_dict['content']
        self.last_full_answer = answer_dict
        return True

    def get_tokens(self, parameters: dict[str, str | int | float]) -> List[int]:
        body_json = json.dumps(parameters)
        connection = http.client.HTTPConnection(f"{self.server_addr}:{self.server_port}")
        connection.request(method="POST",
                url=self.server_tokenizer_endpoint,
                headers=self.post_tokenizer_headers, 
                body=body_json)

        response = connection.getresponse()

        if response.status != 200:
            return False
        
        answer = response.read()
        answer_dict = json.loads(answer)
        return answer_dict['tokens']

class CTransformersProvider(BaseProvider):
    def __init__(self, model_path: str, context_length: int, model_type: str = "llama") -> None:
        super().__init__()
        from ctransformers import AutoModelForCausalLM
        self.model_path = os.path.abspath(model_path)
        self.model_type = model_type
        self.context_length = context_length
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, model_type=self.model_type, context_length=self.context_length)
    
    def query(self, parameters):
        ans = self.model(prompt = parameters['prompt'],
                         temperature = parameters['temperature'],
                         max_new_tokens = parameters['n_predict'])
        
        self.last_answer = ans
        self.last_full_answer = ans
        
        return True
    
    def get_tokens(self, parameters: dict[str, str | int | float]) -> List[int]:
        return self.model.tokenize(parameters['content'])
    
class LLAMACPPProvider(ServerProvider):
    def __init__(self, server_address: str, server_port: str, completion_endpoint: str = "/completion", tokenizer_endpoint: str = "/tokenize", post_completion_headers: str = POST_COMPLETION_HEADERS, post_tokenizer_headers: str = POST_COMPLETION_HEADERS) -> None:
        super().__init__(server_address, server_port, completion_endpoint, tokenizer_endpoint, post_completion_headers, post_tokenizer_headers)