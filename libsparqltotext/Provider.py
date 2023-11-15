from abc import ABC, abstractmethod
import http.client
import json

POST_COMPLETION_HEADERS = {"Content-Type":"application/json"}

class BaseProvider(ABC):
    def __init__(self) -> None:
        self.last_answer = None
        self.last_full_answer = None

    @abstractmethod
    def query(self, parameters):
        pass

    def get_full_answer(self):
        '''Full result from the provider.'''
        return self.last_full_answer

    def get_answer(self):
        '''Only the generated answer.'''
        return self.last_answer

class ServerProvider(BaseProvider):
    def __init__(self, args) -> None:
        super().__init__()
        self.server_addr = args.server_address
        self.server_port = args.server_port
        self.server_completion_endpoint = args.endpoint
        self.post_completion_headers = POST_COMPLETION_HEADERS
    
    def query(self, parameters):
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

class CTransformersProvider(BaseProvider):
    def __init__(self, args) -> None:
        super().__init__()
        from ctransformers import AutoModelForCausalLM
        self.model_path = args.model_path
        self.model_type = "llama"
        self.context_length = args.context_length
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, model_type=self.model_type, context_length=self.context_length)
    
    def query(self, parameters):
        ans = self.model(prompt = parameters['prompt'],
                         temperature = parameters['temperature'],
                         max_new_tokens = parameters['n_predict'])
        
        self.last_answer = ans
        self.last_full_answer = ans