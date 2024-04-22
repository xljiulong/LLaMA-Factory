ps aux | grep "stage dpo" | awk '{print $2}' | xargs -i kill -9 {}

WANDB_DISABLED=1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus 2 --master_port=9527 /workspace/projects/LLaMA-Factory/src/train_bash.py \
    --stage dpo \
    --do_train \
    --deepspeed /workspace/projects/LLaMA-Factory/config/ds_config.json \
    --model_name_or_path /workspace/models/huggingface/chatglm3-6b \
    --adapter_name_or_path /workspace/models/huggingface/chatglm3-6b_exp_sft_lora_llamafactory \
    --create_new_adapter \
    --dataset comparison_gpt4_zh \
    --dataset_dir /workspace/projects/LLaMA-Factory/data \
    --template chatglm3 \
    --finetuning_type lora \
    --lora_target query_key_value \
    --output_dir /workspace/models/huggingface/chatglm3-6b_exp_dpo_lora_llamafactory \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 1e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --fp16
