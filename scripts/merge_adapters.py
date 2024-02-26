from peft import PeftModel
from transformers import AutoModelForCausalLM
import argparse

def merge_model_with_adapters(base_model_name, adapter_model_name, output_file):
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(model, adapter_model_name)

    model = model.merge_and_unload()
    model.save_pretrained(output_file)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Model + Adapter merger",
                                     description="Merge a model with its adapaters.")
    parser.add_argument('-m', '--model-path', required=True, type=str)
    parser.add_argument('-a', '--adapter-path', required=True, type=str)
    parser.add_argument('-o', '--output-path', default="merged_model", type=str)
    
    args = parser.parse_args()
    
    merge_model_with_adapters(args.model_path, args.adapter_path, args.output_path)