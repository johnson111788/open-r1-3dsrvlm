ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml --num_processes=4 src/open_r1/llavagrpo.py --config recipes/llava/llava_next_llama3/grpo/confg_full.yaml

# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml --num_processes=4 src/open_r1/llavagrpo.py --config recipes/llava/llava_v1.5_7b/grpo/confg_full.yaml
