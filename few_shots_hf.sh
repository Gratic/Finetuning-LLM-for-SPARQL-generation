#/bin/bash

python3 scripts/few_shots_hf.py \
 --dataset-name Zaleks/labelized_sparql_wikidata_mistral_v02_q5_limit_10 \
 --dataset-split valid \
 --input-column templated_input \
 --target-column target_template \
 --start-tag "[sparql]" \
 --end-tag "[/sparql]" \
 --base-model "mistralai/Mistral-7B-Instruct-v0.2" \
 --lora-path "path/to/lora/adapter" \
 --max-tokens 100 \
 --temperature 0.7 \
 --n-examples 3 \
 --use-gpu \
 --quantization 4bit \
 --precision fp16