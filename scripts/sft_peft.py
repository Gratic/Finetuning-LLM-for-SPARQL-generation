import os
import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from trl import SFTTrainer
from accelerate import Accelerator
import argparse

tokenizer = None

os.environ["WANDB_PROJECT"] = "SFT_Training test"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

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
    text = f"[INST] Given a question, generate a SPARQL query that answers the question where entities and properties are placeholders. After the generated query, gives the list of placeholders and their corresponding Wikidata identifiers: {example['input']} [/INST] `sparql\n{example['target']}`"
    return text

def main():
    global tokenizer
    
    parser = argparse.ArgumentParser(prog="PEFT (QLora) SFT Script")
    parser.add_argument("-m", "--model", type=str, help="Huggingface model or path to a model to finetune.", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("-trd", "--train-data", required=True, type=str, help="Path to the train dataset.")
    parser.add_argument("-td", "--test-data", required=True, type=str, help="Path to the test dataset.")
    parser.add_argument("-vd", "--valid-data", required=False, type=str, help="Path to the valid dataset.", default="")
    parser.add_argument("-rv", "--rvalue", type=int, help="Lora r-value.", default=8)
    parser.add_argument("-bs", "--batch-size", type=int, help="Batch size for training.", default=1)
    parser.add_argument("-ga", "--gradient-accumulation", type=int, help="Gradient accumulation, number of batch to process before making an optimizer step.", default=4)
    parser.add_argument("-p", "--packing", type=int, help="Train with Packing or not (1=True, 0=False).",  default=1)
    parser.add_argument("-o", "--output", type=str, help="Output directory", default="")
    parser.add_argument("-sn", "--save-name", type=str, help="The folder name where the saved checkpoint will be found.", default="final_checkpoint")
    parser.add_argument("-sa", "--save-adapters", dest='save_adapters', action='store_true', help="Save the adapters.")
    parser.add_argument("-sm", "--save-merged", dest='save_adapters', action='store_true', help="Save the model merged with the adapters.")
    args = parser.parse_args()
    
    datafiles = {
        "train": args.train_data,
        "valid": args.valid_data,
        "test": args.test_data
    }
    
    if args.valid_data == "":
         datafiles = {
            "train": args.train_data,
            "test": args.test_data
        }
         
    do_packing = bool(args.packing)
    
    accelerator = Accelerator()
    dataset = load_dataset("pandas", data_files=datafiles)
    model_id = args.model

    lora_config = LoraConfig(
        r=args.rvalue,
        lora_alpha=args.rvalue*2,
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
        device_map={"": accelerator.process_index}
    )
    
    pretrained_model = prepare_model_for_kbit_training(pretrained_model)

    # https://medium.com/@parikshitsaikia1619/mistral-mastery-fine-tuning-fast-inference-guide-62e163198b06
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if do_packing:
        tokenizer.padding_side = "right"
    # TODO: Create a padding token
    tokenizer.pad_token = tokenizer.unk_token

    pretrained_model.config.pad_token_id = tokenizer.pad_token_id
    
    pretrained_model, tokenizer = accelerator.prepare(pretrained_model, tokenizer)

    print_trainable_parameters(pretrained_model)

    training_args = TrainingArguments(
        bf16=True,
        output_dir=args.output,
        optim="adamw_bnb_8bit",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
        evaluation_strategy="steps",
        eval_steps=0.25,
        logging_strategy="epoch",
        report_to="wandb"
    )

    trainer = SFTTrainer(
        pretrained_model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        formatting_func=format_prompt,
        max_seq_length=4096,
        peft_config=lora_config,
        packing=do_packing,
    )

    trainer.train()
    
    save_path_full = os.path.join(training_args.output_dir, args.save_name)
    save_path_adapters = os.path.join(training_args.output_dir, f"{args.save_name}-adapters")
    
    if args.save_adapters:
        trainer.model.save_pretrained(save_path_adapters)
        
    if args.save_merged:
        trainer.model.merge_and_unload()
        trainer.modelsave_pretrained(save_path_full)

if __name__ == "__main__":
    main()