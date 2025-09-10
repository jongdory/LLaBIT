CUDA_VISIBLE_DEVICES=1 python generate_eval.py \
    LLaBIT__2025-03-25_13-23-38/checkpoint-165498 \
    test_total_f16.pkl \
    outputs \
    --csv_path /store4/01.Database/01.Brain/15.2D/text-split-tr.csv \
    --task i2i \
    --rank 1 \
    --top_p 0.5