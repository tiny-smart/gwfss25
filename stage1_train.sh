export DETECTRON2_DATASETS=data_root
export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 -W ignore train_net.py \
    --config-file configs/gwfss/beit_adapter/maskformer2_beit_adapter_large_bas8_20k.yaml \
    --num-gpus 4 \
    --num-machines 1 \
    SSL.PERCENTAGE 100 \
    SSL.TRAIN_SSL False \
    OUTPUT_DIR outputs/beitv2_sapa_stage1