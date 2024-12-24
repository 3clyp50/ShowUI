#!/bin/bash

export WANDB_API_KEY=<your_wandb_api_key_here>

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 4221 train.py \
  --wandb_key $WANDB_API_KEY \
  --model_id="Qwen/Qwen2-VL-2B-Instruct" \
  --version="Qwen/Qwen2-VL-2B-Instruct" \
  --dataset_dir="./data" \
  --log_base_dir="./logs" \
  --epochs=1 \
  --steps_per_epoch=100 \
  --batch_size=8 \
  --grad_accumulation_steps=4 \
  --model_max_length=4096 \
  --val_dataset="screenspot" \
  --val_omniact_nav_data="hf_test" \
  --exp_id="showui_8x_h100" \
  --sample_rates="1" \
  --dataset="omniact" \
  --omniact_data="hf_train_showui_desktop" \
  --amex_data="hf_train_ele,hf_train_func" \
  --precision="bf16" \
  --attn_imple="flash_attention_2" \
  --workers=16 \
  --lora_r=8 \
  --lora_alpha=64 \
  --min_visual_tokens=256 \
  --max_visual_tokens=1344 \
  --num_history=4 \
  --num_turn=1 \
  --interleaved_history="tttt" \
  --crop_min=0.5 \
  --crop_max=1.5 \
  --random_sample \
  --record_sample \
  --lr=0.0001 \
  --uniform_prompt \
  --ds_zero="zero3" \
  --gradient_checkpointing
