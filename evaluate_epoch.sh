python3 scripts/evaluation_bench.py \
  --dataset outputs/batch_run/nt_basic_vs_template/models/M-7B-I-v0.2_optawbn8-cdfp16-lr000005-rv512-ramul2-ld005-bs2-ga8-gc1-p0-nta0-e3-ctx3072-q4bit-template-template-stsparql_compute_metrics_0.json \
  --generated-field generated_texts \
  --executed-field "" \
  --hf-dataset Zaleks/labelized_sparql_wikidata_mistral_v02_q5_limit_10 \
  --hf-split valid \
  --hf-target target_template \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --output ./outputs/epoch_evaluation/ \
  --save-name mistral_template_template_valid_epoch_1 \
  --execute-queries \
  --start-tag "[sparql]" \
  --end-tag "[/sparql]"

  python3 scripts/evaluation_bench.py \
  --dataset outputs/batch_run/nt_basic_vs_template/models/M-7B-I-v0.2_optawbn8-cdfp16-lr000005-rv512-ramul2-ld005-bs2-ga8-gc1-p0-nta0-e3-ctx3072-q4bit-template-template-stsparql_compute_metrics_1.json \
  --generated-field generated_texts \
  --executed-field "" \
  --hf-dataset Zaleks/labelized_sparql_wikidata_mistral_v02_q5_limit_10 \
  --hf-split valid \
  --hf-target target_template \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --output ./outputs/epoch_evaluation/ \
  --save-name mistral_template_template_valid_epoch_2 \
  --execute-queries \
  --start-tag "[sparql]" \
  --end-tag "[/sparql]"