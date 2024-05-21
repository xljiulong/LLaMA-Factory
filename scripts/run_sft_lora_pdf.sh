WANDB_DISABLED=1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus 2 --master_port=9527 /workspace/projects/LLaMA-Factory/src/train_bash.py \
    --stage sft \
    --do_train \
    --deepspeed /workspace/projects/LLaMA-Factory/examples/deepspeed/ds_z3_config.json \
    --model_name_or_path /workspace/models/huggingface/chatglm2-6b-32k \
    --dataset pdf_corpus \
    --dataset_dir /workspace/projects/LLaMA-Factory/data \
    --template chatglm3 \
    --finetuning_type lora \
    --lora_target query_key_value	 \
    --output_dir /workspace/models/huggingface/chatglm32k_exp_pdf_sft_lora_llamafactory \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 32000 \
    --preprocessing_num_workers 8 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 100 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --num_train_epochs 6.0 \
    --max_samples 8000 \
    --val_size 0.1 \
    --plot_loss \
    --fp16
