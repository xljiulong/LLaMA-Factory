# ps aux | grep "train_bash" | awk '{print $2}' | xargs -i kill -9 {}
export CUDA_VISIBLE_DEVICES=1

FORCE_TORCHRUN=1 python /workspace/projects/LLaMA-Factory/src/llamafactory/cli.py export  /workspace/projects/LLaMA-Factory/examples/merge_lora/qwen2_lora_sft.yaml