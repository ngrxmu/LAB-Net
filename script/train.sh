#!/bin/bash
batchs=2 # ISTD 2 SRD 1
DISPLAY_PORT=${2:-7000}
GPU=$1
lr=0.0002
loadSize=256 # ISTD 256 SRD 400
fineSize=256 # ISTD 256 SRD 400
L1=100
down_w=256 # ISTD 256 SRD 128
down_h=256 # ISTD 256 SRD 128
model=LAB
G='LABNet'
checkpoint='checkpoints/'
datasetmode="shadowgt"
dataroot='' # Need to be specify before training
name='' # Need to be specify before training

NAME="${name}"

OTHER="--save_epoch_freq 100 --niter 50 --niter_decay 250  --test_epoch_freq 15"

CMD="python ../train.py --loadSize ${loadSize} \
    --name ${NAME} \
    --dataroot  ${dataroot}\
    --checkpoints_dir ${checkpoint} \
    --down_w $down_w     --down_h $down_h \
    --fineSize $fineSize --model $model\
    --batch_size $batchs --display_port ${DISPLAY_PORT} --display_server http://localhost\
    --phase train_  --gpu_ids ${GPU} --lr ${lr} \
    --randomSize --keep_ratio \
    --lambda_L1 ${L1} \
    --dataset_mode $datasetmode \
    --netG ${G} \
    --no_html\
    $OTHER
"
echo $CMD
eval $CMD

