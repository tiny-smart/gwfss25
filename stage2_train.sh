export DETECTRON2_DATASETS=data_root
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -W ignore train_net.py \
    --config-file configs/gwfss/beit_adapter/maskformer2_beit_adapter_large_50k_stage2.yaml \
    --num-gpus 4 \
    --num-machines 1 \
    SSL.PERCENTAGE 100 \
    SSL.TRAIN_SSL True \
    SSL.TEACHER_CKPT outputs/stage1_teacher_sapa_conv_dim256/model_iter0014999.pth \
    SSL.BURNIN_ITER 20000 \
    OUTPUT_DIR outputs/stage2_beitv2_sapa