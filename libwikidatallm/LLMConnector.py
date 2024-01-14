from abc import ABC, abstractmethod
from typing import List
import json
import os

POST_COMPLETION_HEADERS = {"Content-Type":"application/json"}

class LLMResponse():
    def __init__(self, full_answer, generated_text: str) -> None:
        self.full_answer = full_answer
        self.generated_text = generated_text

class LLMConnector(ABC):
    @abstractmethod
    def completion(self, prompt: str) -> LLMResponse:
        pass
    
    @abstractmethod
    def tokenize(self, prompt: str) -> List[int]:
        pass
    
    def set_config(self, temperature: float, max_number_of_tokens_to_generate: int):
        self.temperature = temperature
        self.max_number_of_tokens_to_generate = max_number_of_tokens_to_generate
        return self
    
class ServerConnector(LLMConnector):
    def __init__(self, server_address: str = "", server_port: str = "", completion_endpoint: str = "", tokenizer_endpoint: str = "", post_completion_headers: str = POST_COMPLETION_HEADERS, post_tokenizer_headers: str = POST_COMPLETION_HEADERS, temperature: float = 0.2, max_number_of_tokens_to_generate: int = 256) -> None:
        self.create_connection(server_address, server_port, completion_endpoint, tokenizer_endpoint, post_completion_headers, post_tokenizer_headers)
        self.set_config(temperature=temperature, max_number_of_tokens_to_generate=max_number_of_tokens_to_generate)
        
    def create_connection(self, server_address: str, server_port: str, completion_endpoint: str, tokenizer_endpoint: str, post_completion_headers: str = POST_COMPLETION_HEADERS, post_tokenizer_headers: str = POST_COMPLETION_HEADERS):
        import http.client
        self.server_address = server_address
        self.server_port = server_port
        self.completion_endpoint = completion_endpoint
        self.tokenizer_endpoint = tokenizer_endpoint
        self.post_completion_headers = post_completion_headers
        self.post_tokenizer_headers = post_tokenizer_headers
        
        self.httpconnection = http.client.HTTPConnection(f"{self.server_address}:{self.server_port}")
        return self
    
    def completion(self, prompt: str) -> LLMResponse:
        parameters = self.create_payload_completion(prompt)
        body_json = json.dumps(parameters)
        
        self.httpconnection.request(method="POST",
                url=self.completion_endpoint,
                headers=self.post_completion_headers, 
                body=body_json)

        response = json.loads(self.httpconnection.getresponse().read())
        
        return LLMResponse(response, response['content'])
    
    def tokenize(self, prompt: str) -> List[int]:
        parameters = self.create_payload_tokenize(prompt)
        body_json = json.dumps(parameters)
        
        self.httpconnection.request(method="POST",
                url=self.tokenizer_endpoint,
                headers=self.post_tokenizer_headers, 
                body=body_json)

        response = json.loads(self.httpconnection.getresponse().read())

        return response['tokens']

    def create_payload_completion(self, prompt):
        return {
            "temperature": self.temperature,
            "n_predict": self.max_number_of_tokens_to_generate,
            "prompt": prompt 
        }
        
    def create_payload_tokenize(self, prompt):
        return {
            "content": prompt
        }
    
class LlamaCPPConnector(ServerConnector):
    def __init__(self, server_address: str = "127.0.0.1", server_port: str = "8080", completion_endpoint: str = "/completion", tokenizer_endpoint: str = "/tokenize", post_completion_headers: str = POST_COMPLETION_HEADERS, post_tokenizer_headers: str = POST_COMPLETION_HEADERS, temperature: float = 0.2, max_number_of_tokens_to_generate: int = 256) -> None:
        super().__init__(server_address, server_port, completion_endpoint, tokenizer_endpoint, post_completion_headers, post_tokenizer_headers, temperature, max_number_of_tokens_to_generate)
        
class CTransformersConnector(LLMConnector):
    def __init__(self, model_path: str, context_length: int, model_type: str = "llama", temperature: float = 0.2, max_number_of_tokens_to_generate: int = 256) -> None:
        super().__init__()
        from ctransformers import AutoModelForCausalLM
        self.model_path = os.path.abspath(model_path)
        self.model_type = model_type
        self.context_length = context_length
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, model_type=self.model_type, context_length=self.context_length)
        
        self.set_config(temperature, max_number_of_tokens_to_generate)
    
    def completion(self, prompt: str):
        ans = self.model(prompt = prompt,
                         temperature = self.temperature,
                         max_new_tokens = self.max_number_of_tokens_to_generate)
        
        return LLMResponse(ans, ans)
    
    def tokenize(self, prompt: str) -> List[int]:
        return self.model.tokenize(prompt)

class vLLMConnector(LLMConnector):
    def __init__(self, model_path: str, tokenizer: str, context_length: int, temperature: float = 0.2, top_p: float = 0.95, max_number_of_tokens_to_generate: int = 256) -> None:
        super().__init__()
        from vllm import LLM, SamplingParams
        self.model_path = os.path.abspath(model_path)
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.temperature = temperature
        self.top_p = top_p
        self.max_number_of_tokens_to_generate = max_number_of_tokens_to_generate
        
        self.sampling_params = SamplingParams(temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_number_of_tokens_to_generate)
        self.model = LLM(model=self.model_path, tokenizer=self.tokenizer)
    
    def completion(self, prompt: str) -> LLMResponse:
        outputs = self.model.generate(prompt, self.sampling_params, use_tqdm=False)
        output = outputs[0]
        generated_text = output.outputs[0].text
        return LLMResponse(output, generated_text)
    
    def tokenize(self, prompt: str) -> List[int]:
        return self.model.get_tokenizer().encode(prompt)

    def batch_completion(self, prompts: str) -> List[LLMResponse]:
        outputs = self.model.generate(prompts, self.sampling_params, use_tqdm=False)
        
        responses = []
        
        for output in outputs:
            generated_text = output.outputs[0].text
            responses.append(LLMResponse(output, generated_text))
        
        return responses