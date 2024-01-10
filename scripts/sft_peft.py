import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from trl import SFTTrainer
from accelerate import Accelerator
import os

tokenizer = None

# https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/supervised_finetuning.py
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def format_prompt(example):
    prefix = ""
    
    text = f"{tokenizer.bos_token}[INST] Answer this question with a SPARQL query: {example['input']} [/INST] {example['target']}{tokenizer.eos_token}"
    return text

def main():
    global tokenizer
    
    dataset = load_dataset("pandas", data_files="./outputs/finetune_dataset.pkl")
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    pretrained_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": Accelerator().process_index}
    )

    # https://medium.com/@parikshitsaikia1619/mistral-mastery-fine-tuning-fast-inference-guide-62e163198b06
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.unk_token

    pretrained_model.config.pad_token_id = tokenizer.pad_token_id

    print_trainable_parameters(pretrained_model)

    training_args = TrainingArguments(
        bf16=True,
        output_dir="./outputs/training/",
        optim="adamw_bnb_8bit",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        pretrained_model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        formatting_func=format_prompt,
        max_seq_length=4096,
        peft_config=lora_config,
        packing=True,
    )

    trainer.train()
    
    trainer.model.save_pretrained(os.path.join(training_args.output_dir, "final_checkpoint/"))
