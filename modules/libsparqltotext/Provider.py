from abc import ABC, abstractmethod
import http.client
import json
import os
from typing import List, Dict, Union
import openai
import time
from functools import wraps

POST_COMPLETION_HEADERS = {"Content-Type":"application/json"}

def retry_after(seconds, max_attempts=3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        raise e
                    print(f"Attempt {attempts} failed. Retrying in {seconds} seconds...")
                    time.sleep(seconds)
        return wrapper
    return decorator

class BaseProvider(ABC):
    def __init__(self) -> None:
        self.last_answer: str = None
        self.last_full_answer: Union[str, Dict[str, str]] = None

    @abstractmethod
    def query(self, prompt: str) -> bool:
        pass

    def get_full_answer(self):
        '''Full result from the provider.'''
        return self.last_full_answer

    def get_answer(self):
        '''Only the generated answer.'''
        return self.last_answer

    @abstractmethod
    def get_tokens(self, prompt: str) -> List[int]:
        pass

class ServerProvider(BaseProvider):
    def __init__(self, server_address: str,
                server_port: str,
                completion_endpoint: str,
                tokenizer_endpoint: str,
                post_completion_headers: str = POST_COMPLETION_HEADERS,
                post_tokenizer_headers: str = POST_COMPLETION_HEADERS,
                temperature: float = 0.2,
                n_predict: int = 256
                ) -> None:
        super().__init__()
        self.server_addr = server_address
        self.server_port = server_port
        self.server_completion_endpoint = completion_endpoint
        self.server_tokenizer_endpoint = tokenizer_endpoint
        self.post_completion_headers = post_completion_headers
        self.post_tokenizer_headers = post_tokenizer_headers
        self.temperature = temperature
        self.n_predict = n_predict
    
    def query(self, prompt: str) -> bool:
        body_json = json.dumps({
            "prompt": prompt,
            "temperature": self.temperature,
            "n_predict": self.n_predict
        })
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

    def get_tokens(self, prompt: str) -> List[int]:
        body_json = json.dumps({
            "content": prompt
        })
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
    def __init__(self, model_path: str, context_length: int, model_type: str = "llama", temperature: float = 0.2, n_predict: int = 256) -> None:
        super().__init__()
        from ctransformers import AutoModelForCausalLM
        self.model_path = model_path
        self.model_type = model_type
        self.context_length = context_length
        self.temperature = temperature
        self.n_predict = n_predict
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, model_type=self.model_type, context_length=self.context_length)
    
    def query(self, prompt: str):
        ans = self.model(prompt = prompt,
                         temperature = self.temperature,
                         max_new_tokens = self.n_predict)
        
        self.last_answer = ans
        self.last_full_answer = ans
        
        return True
    
    def get_tokens(self, prompt: str) -> List[int]:
        return self.model.tokenize(prompt)
    
class LLAMACPPProvider(ServerProvider):
    def __init__(self, server_address: str, server_port: str, completion_endpoint: str = "/completion", tokenizer_endpoint: str = "/tokenize", post_completion_headers: str = POST_COMPLETION_HEADERS, post_tokenizer_headers: str = POST_COMPLETION_HEADERS, temperature: float = 0.2, n_predict: int = 256) -> None:
        super().__init__(server_address, server_port, completion_endpoint, tokenizer_endpoint, post_completion_headers, post_tokenizer_headers, temperature, n_predict)

class vLLMProvider(BaseProvider):
    def __init__(self, model_path: str, context_length: int, temperature: float = 0.2, n_predict: int = 256, top_p: float = 0.95) -> None:
        super().__init__()
        from vllm import LLM, SamplingParams
        self.model_path = model_path
        self.temperature = temperature
        self.n_predict = n_predict
        self.top_p = top_p
        self.tokenizer = model_path
        self.context_length = context_length
        self.sampling_params = SamplingParams(temperature=self.temperature, top_p=self.top_p, max_tokens=self.n_predict)
        
        self.model = LLM(model=self.model_path, tokenizer=self.tokenizer)
    
    def query(self, prompt: str) -> bool:
        outputs = self.model.generate(prompt, self.sampling_params, use_tqdm=False)
        output = outputs[0]
        generated_text = output.outputs[0].text
        
        self.last_answer = generated_text
        self.last_full_answer = output
        
        return True
    
    def get_tokens(self, prompt: str) -> List[int]:
        return self.model.get_tokenizer().encode(prompt)
    
class TransformersProvider(BaseProvider):
    def __init__(self, model_path: str, context_length: int, temperature: float = 0.2, n_predict: int = 256, top_p: float = 0.95) -> None:
        super().__init__()
        import torch
        from transformers import AutoModelForCausalLM, GenerationConfig, AutoTokenizer
        self.model_path = model_path
        self.context_length = context_length
        self.temperature = temperature
        self.n_predict = n_predict
        self.top_p = top_p
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        self.config = GenerationConfig(
            do_sample = True,
            temperature = self.temperature,
            top_p = self.top_p,
            max_new_tokens = self.n_predict,
            eos_token_id = self.tokenizer.eos_token_id,
            pad_token_id = self.tokenizer.pad_token_id,
            )
    
    def query(self, prompt: str) -> bool:
        self.model.eval()
        
        inputs = self.tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(self.device)
        outputs = self.model.generate(**inputs, generation_config=self.config)
        decoded_outputs = self.tokenizer.decode(outputs.squeeze())
        
        self.last_answer = decoded_outputs
        self.last_full_answer = decoded_outputs
        
        return True
    
    def get_tokens(self, prompt: str) -> List[int]:
        return self.tokenizer.encode(prompt)
    
class TransformersProviderv2(BaseProvider):
    def __init__(self, model_path: str, context_length: int, temperature: float = 0.2, n_predict: int = 256, top_p: float = 0.95) -> None:
        super().__init__()
        import torch
        from transformers import AutoModelForCausalLM, GenerationConfig, AutoTokenizer, Pipeline
        self.model_path = model_path
        self.context_length = context_length
        self.temperature = temperature
        self.n_predict = n_predict
        self.top_p = top_p
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        self.config = GenerationConfig(
            do_sample = True,
            temperature = self.temperature,
            top_p = self.top_p,
            max_new_tokens = self.n_predict,
            eos_token_id = self.tokenizer.eos_token_id,
            pad_token_id = self.tokenizer.pad_token_id,
            )
        
        self.pipeline = Pipeline(
            model = self.model,
            tokenizer= self.tokenizer,
            generation_config = self.config,
            device = self.device,
            framework="pt"
        )
    
    def query(self, prompt: str) -> bool:
        self.model.eval()
        
        outputs = self.pipeline(inputs=prompt)
        output = outputs[0]
        generated_text = output['generated_text']
        
        self.last_answer = generated_text
        self.last_full_answer = outputs
        
        return True
    
    def get_tokens(self, prompt: str) -> List[int]:
        return self.tokenizer.encode(prompt)
    
class OpenAIProvider(BaseProvider):
    def __init__(self, model_name:str, api_key:str, base_url:str = "https://llms-inference.innkube.fim.uni-passau.de", temperature: float = 0.2, n_predict: int = 256, top_p: float = 0.95) -> None:
        super().__init__()
        
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.n_predict = n_predict
        self.top_p = top_p
        
        self.client = openai.OpenAI(
        api_key = self.api_key,
        base_url= base_url)
    
    @retry_after(seconds=120, max_attempts=5)
    def query(self, prompt: str) -> bool:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.n_predict
        )
        
        self.last_answer = response.choices[0].message.content
        self.last_full_answer = response
        
        return True
    
    def get_tokens(self, prompt: str) -> List[int]:
        # Heuristic because no tokenizer API

        return list(range(len(prompt.split(" "))*2))