# pip install -q transformers
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "bigscience/bloomz-560m"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")

prompt = "Is Chinchilla better? \"The paper shows that LLMs are undertrained. It demonstrates that with more training tokens and fewer parameters, they achieve superior performance compared to current training methods. Chinchilla has 70B parameters and is trained on 1.4T tokens. Gopher has 280B parameters and is trained on 300B tokens. With an equivalent number of FLOPs, Chinchilla improves Gopher's score on the MMLU benchmark by 7.\""

inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(inputs, max_length=len(prompt) + 100)
print(tokenizer.decode(outputs[0]))

# generator = pipeline("")