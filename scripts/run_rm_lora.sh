ps aux | grep "train_bash" | awk '{print $2}' | xargs -i kill -9 {}
export CUDA_VISIBLE_DEVICES=0

WANDB_DISABLED=1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus 1 --master_port=9527 /workspace/projects/LLaMA-Factory/src/train_bash.py \
    --stage rm \
    --do_train \
    --deepspeed /workspace/projects/LLaMA-Factory/examples/deepspeed/ds_z3_offload_config.json \
    --model_name_or_path /workspace/models/huggingface/chatglm3-6b \
    --adapter_name_or_path /workspace/models/huggingface/chatglm32k_exp_sft_lora_llamafactory \
    --create_new_adapter \
    --dataset comparison_gpt4_zh \
    --dataset_dir /workspace/projects/LLaMA-Factory/data \
    --template chatglm3 \
    --finetuning_type lora \
    --lora_target query_key_value \
    --output_dir /workspace/models/huggingface/chatglm32k_rm_sft_lora_llamafactory \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 4 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 5 \
    --eval_steps 20 \
    --evaluation_strategy steps \
    --learning_rate 1e-5 \
    --num_train_epochs 2.0 \
    --max_samples 5000 \
    --val_size 0.1 \
    --plot_loss \
    --save_safetensors False \
    --fp16
