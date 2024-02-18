from abc import ABC, abstractmethod
import http.client
import json
import os
from typing import List, Dict, Union

POST_COMPLETION_HEADERS = {"Content-Type":"application/json"}

class BaseProvider(ABC):
    def __init__(self) -> None:
        self.last_answer: str = None
        self.last_full_answer: Union[str, Dict[str, str]] = None

    @abstractmethod
    def query(self, parameters: Dict[str, "str | int | float"]) -> bool:
        pass

    def get_full_answer(self):
        '''Full result from the provider.'''
        return self.last_full_answer

    def get_answer(self):
        '''Only the generated answer.'''
        return self.last_answer

    @abstractmethod
    def get_tokens(self, parameters: Dict[str, "str | int | float"]) -> List[int]:
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
    
    def query(self, parameters: Dict[str, "str | int | float"]) -> bool:
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

    def get_tokens(self, parameters: Dict[str, "str | int | float"]) -> List[int]:
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
        self.model_path = model_path
        self.model_type = model_type
        self.context_length = context_length
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, model_type=self.model_type, context_length=self.context_length)
    
    def query(self, parameters: Dict[str, Union[str, int, float]]):
        ans = self.model(prompt = parameters['prompt'],
                         temperature = parameters['temperature'],
                         max_new_tokens = parameters['n_predict'])
        
        self.last_answer = ans
        self.last_full_answer = ans
        
        return True
    
    def get_tokens(self, parameters: Dict[str, Union[str, int, float]]) -> List[int]:
        return self.model.tokenize(parameters['content'])
    
class LLAMACPPProvider(ServerProvider):
    def __init__(self, server_address: str, server_port: str, completion_endpoint: str = "/completion", tokenizer_endpoint: str = "/tokenize", post_completion_headers: str = POST_COMPLETION_HEADERS, post_tokenizer_headers: str = POST_COMPLETION_HEADERS) -> None:
        super().__init__(server_address, server_port, completion_endpoint, tokenizer_endpoint, post_completion_headers, post_tokenizer_headers)

class vLLMProvider(BaseProvider):
    def __init__(self, model_path: str, context_length: int) -> None:
        super().__init__()
        from vllm import LLM
        self.model_path = model_path
        self.tokenizer = model_path
        self.context_length = context_length
        
        self.model = LLM(model=self.model_path, tokenizer=self.tokenizer)
    
    def query(self, parameters: Dict[str, Union[str, int, float]]) -> bool:
        from vllm import SamplingParams
        sampling_params = SamplingParams(temperature=parameters['temperature'], top_p=0.95, max_tokens=parameters['n_predict'])
        outputs = self.model.generate(parameters['prompt'], sampling_params, use_tqdm=False)
        output = outputs[0]
        generated_text = output.outputs[0].text
        
        self.last_answer = generated_text
        self.last_full_answer = output
        
        return True
    
    def get_tokens(self, parameters: Dict[str, Union[str, int, float]]) -> List[int]:
        return self.model.get_tokenizer().encode(parameters['content'])

class vLLMProvider(BaseProvider):
    def __init__(self, model_path: str, context_length: int) -> None:
        super().__init__()
        from vllm import LLM
        self.model_path = model_path
        self.tokenizer = model_path
        self.context_length = context_length
        
        self.model = LLM(model=self.model_path, tokenizer=self.tokenizer)
    
    def query(self, parameters: Dict[str, Union[str, int, float]]) -> bool:
        from vllm import SamplingParams
        sampling_params = SamplingParams(temperature=parameters['temperature'], top_p=0.95, max_tokens=parameters['n_predict'])
        outputs = self.model.generate(parameters['prompt'], sampling_params, use_tqdm=False)
        output = outputs[0]
        generated_text = output.outputs[0].text
        
        self.last_answer = generated_text
        self.last_full_answer = output
        
        return True
    
    def get_tokens(self, parameters: Dict[str, Union[str, int, float]]) -> List[int]:
        return self.model.get_tokenizer().encode(parameters['content'])
    
class TransformersProvider(BaseProvider):
    def __init__(self, model_path: str, context_length: int, top_p: float = 0.95) -> None:
        super().__init__()
        import torch
        from transformers import AutoModelForCausalLM, GenerationConfig, AutoTokenizer
        self.model_path = model_path
        self.context_length = context_length
        self.top_p = top_p
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # self.pipeline = Pipeline(
        #     "text-generation",
        #     model = self.model,
        #     tokenizer= self.tokenizer,
        #     generation_config = self.config,
        #     device = self.device,
        #     framework="pt"
        # )
        
    def query(self, parameters: Dict[str, Union[str, int, float]]) -> bool:
        from transformers import GenerationConfig
        self.model.eval()
        
        self.config = GenerationConfig(
            do_sample = True,
            temperature = parameters['temperature'],
            top_p = self.top_p,
            max_new_tokens = parameters['n_predict'],
            eos_token_id = self.tokenizer.eos_token_id,
            pad_token_id = self.tokenizer.pad_token_id,
            )
        
        # outputs = self.pipeline(inputs=prompt)
        inputs = self.tokenizer([parameters['prompt']], return_tensors="pt")
        inputs = inputs.to(self.device)
        outputs = self.model.generate(**inputs, generation_config=self.config)
        decoded_outputs = self.tokenizer.decode(outputs.squeeze())
        # output = outputs[0]
        # generated_text = output['generated_text']
        
        self.last_answer = decoded_outputs
        self.last_full_answer = decoded_outputs
        
        return True
    
    def tokenize(self, parameters: Dict[str, Union[str, int, float]]) -> List[int]:
        return self.tokenizer.encode(parameters['content'])