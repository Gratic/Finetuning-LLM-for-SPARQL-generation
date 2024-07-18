#/bin/bash

python3 scripts/few_shots_hf.py \
 --dataset-name Zaleks/labelized_sparql_wikidata_mistral_v02_q5_limit_10 \
 --dataset-split valid \
 --input-column templated_input \
 --target-column target_template \
 --start-tag "[sparql]" \
 --end-tag "[/sparql]" \
 --base-model "mistralai/Mistral-7B-Instruct-v0.2" \
 --lora-path "outputs/batch_run/nt_basic_vs_template/models/M-7B-I-v0.2_optawbn8-cdfp16-lr000005-rv512-ramul2-ld005-bs2-ga8-gc1-p0-nta0-e3-ctx3072-q4bit-template-template-stsparql_adapters" \
 --max-tokens 1024 \
 --temperature 0.2 \
 --n-examples 0 \
 --use-gpu \
 --quantization 4bit \
 --precision fp16 \
 --output "outputs/few_shots_hf/bvt_mistral" \
 --save-name "rv512-template-template-t02-q4-fp16-zeroshot"

python3 scripts/few_shots_hf.py \
 --dataset-name Zaleks/labelized_sparql_wikidata_mistral_v02_q5_limit_10 \
 --dataset-split valid \
 --input-column templated_input \
 --target-column target_template \
 --start-tag "[sparql]" \
 --end-tag "[/sparql]" \
 --base-model "mistralai/Mistral-7B-Instruct-v0.2" \
 --lora-path "outputs/batch_run/nt_basic_vs_template/models/M-7B-I-v0.2_optawbn8-cdfp16-lr000005-rv512-ramul2-ld005-bs2-ga8-gc1-p0-nta0-e3-ctx3072-q4bit-template-template-stsparql_adapters" \
 --max-tokens 1024 \
 --temperature 0.2 \
 --n-examples 0 \
 --use-gpu \
 --quantization no \
 --precision fp32 \
 --output "outputs/few_shots_hf/bvt_mistral" \
 --save-name "rv512-template-template-t02-noq-fp32-zeroshot"

python3 scripts/few_shots_hf.py \
 --dataset-name Zaleks/labelized_sparql_wikidata_mistral_v02_q5_limit_10 \
 --dataset-split valid \
 --input-column templated_input \
 --target-column target_template \
 --start-tag "[sparql]" \
 --end-tag "[/sparql]" \
 --base-model "mistralai/Mistral-7B-Instruct-v0.2" \
 --lora-path "outputs/batch_run/nt_basic_vs_template/models/M-7B-I-v0.2_optawbn8-cdfp16-lr000005-rv512-ramul2-ld005-bs2-ga8-gc1-p0-nta0-e3-ctx3072-q4bit-template-template-stsparql_adapters" \
 --max-tokens 1024 \
 --temperature 0.0 \
 --n-examples 0 \
 --use-gpu \
 --quantization no \
 --precision fp32 \
 --output "outputs/few_shots_hf/bvt_mistral" \
 --save-name "rv512-template-template-t00-noq-fp32-zeroshot"

python3 scripts/few_shots_hf.py \
 --dataset-name Zaleks/labelized_sparql_wikidata_mistral_v02_q5_limit_10 \
 --dataset-split valid \
 --input-column templated_input \
 --target-column target_template \
 --start-tag "[sparql]" \
 --end-tag "[/sparql]" \
 --base-model "mistralai/Mistral-7B-Instruct-v0.2" \
 --lora-path "outputs/batch_run/nt_basic_vs_template/models/M-7B-I-v0.2_optawbn8-cdfp16-lr000005-rv512-ramul2-ld005-bs2-ga8-gc1-p0-nta0-e3-ctx3072-q4bit-template-template-stsparql_adapters" \
 --max-tokens 1024 \
 --temperature 0.0 \
 --n-examples 0 \
 --use-gpu \
 --quantization 4bit \
 --precision fp16 \
 --output "outputs/few_shots_hf/bvt_mistral" \
 --save-name "rv512-template-template-t00-q4-fp16-zeroshot"

python3 scripts/few_shots_hf.py \
 --dataset-name Zaleks/labelized_sparql_wikidata_mistral_v02_q5_limit_10 \
 --dataset-split valid \
 --input-column templated_input \
 --target-column target_template \
 --start-tag "[sparql]" \
 --end-tag "[/sparql]" \
 --base-model "mistralai/Mistral-7B-Instruct-v0.2" \
 --lora-path "outputs/batch_run/nt_basic_vs_template/models/M-7B-I-v0.2_optawbn8-cdfp16-lr000005-rv512-ramul2-ld005-bs2-ga8-gc1-p0-nta0-e3-ctx3072-q4bit-template-template-stsparql_adapters" \
 --max-tokens 1024 \
 --temperature 0.0 \
 --n-examples 3 \
 --use-gpu \
 --quantization 4bit \
 --precision fp16 \
 --output "outputs/few_shots_hf/bvt_mistral" \
 --save-name "rv512-template-template-t00-q4-fp16-threeshot"