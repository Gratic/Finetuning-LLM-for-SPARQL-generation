from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, AutoModelForCausalLMWithValueHead

def prompt_generator(example):
    output_texts = []
    batch_size = len(example['input'])
    for i in range(batch_size):
        for j in range(len(example['input'][i])):
            prefix = "Answer this question with a SPARQL query:\n"
            
            text = f"{prefix}{example['input'][i][j]}\n{example['target'][i]}"
            output_texts.append(text)
    return output_texts

model_id = "mistralai/Mistral-7B-Instruct-v0.2"

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

pretrained_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_id,
    peft_config=lora_config,
    load_in_4bit=True,
    load_in_8bit=False
)

dataset = load_dataset("pandas", data_files="./outputs/finetune_dataset.pkl")

trainer = SFTTrainer(
    pretrained_model,
    train_dataset=dataset["train"],
    formatting_func=prompt_generator,
    max_seq_length=4096,
    peft_config=lora_config,
    packing=True
)

trainer.train()