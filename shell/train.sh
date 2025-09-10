#!/bin/bash
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0

export timestamp=`date +%Y-%m-%d_%H-%M-%S`
export model_name='LLaBIT'
export checkpoint_dir_name="${model_name}__${timestamp}"
export deepspeed_config=`pwd`/config/ds_z3_bf16_config.json
export local_training_root='./'
export local_output_dir="${local_training_root}/${checkpoint_dir_name}"
export dbfs_output_dir=''
export tensorboard_display_dir="${local_output_dir}/runs"
export input_model="LLaBIT__2025-03-25_13-23-38/checkpoint-165498" #"databricks/dolly-v2-3b"

deepspeed  \
     --include=localhost:1 \
     --module training.trainer \
     --input-model $input_model \
     --deepspeed $deepspeed_config \
     --epochs 5 \
     --local-output-dir $local_output_dir \
     --per-device-train-batch-size 2 \
     --per-device-eval-batch-size 1 \
     --logging-steps 50 \
     --save-total-limit 5 \
     --eval-steps 50000 \
     --warmup-steps 50 \
     --test-size 200 \
     --lr 5e-6 
     