#/bin/bash

python3 scripts/few_shots.py --api-key sk-3dLKM0b0_vuNjFsy1wZb8w --base-url https://llms-inference.innkube.fim.uni-passau.de \
--dataset-name 'Zaleks/labelized_sparql_wikidata_llama3_70B_limit_10' --dataset-split valid \
--input-column 'basic_input' --target-column 'target_raw' \
--start-tag '```sparql' --end-tag '```' \
--model llama3 --max-tokens 1024 --temperature 0 --n-examples 3

python3 scripts/few_shots.py --api-key sk-3dLKM0b0_vuNjFsy1wZb8w --base-url https://llms-inference.innkube.fim.uni-passau.de \
--dataset-name 'Zaleks/labelized_sparql_wikidata_llama3_70B_limit_10' --dataset-split valid \
--input-column 'templated_input' --target-column 'target_raw' \
--start-tag '```sparql' --end-tag '```' \
--model llama3 --max-tokens 1024 --temperature 0 --n-examples 3

python3 scripts/few_shots.py --api-key sk-3dLKM0b0_vuNjFsy1wZb8w --base-url https://llms-inference.innkube.fim.uni-passau.de \
--dataset-name 'Zaleks/labelized_sparql_wikidata_llama3_70B_limit_10' --dataset-split valid \
--input-column 'basic_input' --target-column 'target_template' \
--start-tag '```sparql' --end-tag '```' \
--model llama3 --max-tokens 1024 --temperature 0 --n-examples 3

python3 scripts/few_shots.py --api-key sk-3dLKM0b0_vuNjFsy1wZb8w --base-url https://llms-inference.innkube.fim.uni-passau.de \
--dataset-name 'Zaleks/labelized_sparql_wikidata_llama3_70B_limit_10' --dataset-split valid \
--input-column 'templated_input' --target-column 'target_template' \
--start-tag '```sparql' --end-tag '```' \
--model llama3 --max-tokens 1024 --temperature 0 --n-examples 3