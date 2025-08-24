python3 -W ignore inference.py \
    --config-file configs/gwfss/beit_adapter/maskformer2_beit_adapter_large_50k_stage2.yaml \
    --num-gpus 1 \
    --num-machines 1 \
    --resume \
    --eval-only \
    SSL.EVAL_WHO teacher
