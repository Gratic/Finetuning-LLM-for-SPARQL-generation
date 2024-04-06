import unittest
from modules.libwikidatallm.LLMConnector import PeftConnector
import torch
from parameterized import parameterized

class PeftConnectorTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
    
    # This test is subject to VRAM and can fail if not enough VRAM is available.
    # The model adapters are specifically for Mistral 7B Instruct v0.2.
    # Changing the model, requires new adapters to work.
    @parameterized.expand(['fp32', 'fp16', 'bf16'])
    def test_model_is_loaded_with_right_computational_dtype_param(self, dtype):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if dtype == "bf16" and not torch.cuda.is_bf16_supported():
            return

        llm = PeftConnector(
            model_path="mistralai/Mistral-7B-Instruct-v0.2",
            adapter_path="tests/libwikidatallmtests/data/llmconnector_adapter_test",
            context_length=512,
            dtype=dtype,
            decoding_strategy="greedy",
            max_number_of_tokens_to_generate=512,
        )
        
        expected_dtype = torch.float32
        if dtype == "fp16": expected_dtype = torch.float16
        if dtype == "bf16": expected_dtype = torch.bfloat16
        
        for param in llm.model.parameters():
            self.assertIsInstance(param.dtype, expected_dtype)
            self.assertIsInstance(param.device, torch.device(device))