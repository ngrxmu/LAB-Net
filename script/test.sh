#!/bin/bash
batchs=1
GPU=$1
lr=0.0002
loadSize=256
fineSize=256
L1=100
size_w=640 # ISTD 640 SRD 840
size_h=480 # ISTD 480 SRD 640
down_w=256 # ISTD 256 SRD 128
down_h=256 # ISTD 256 SRD 128
model=LAB
G='LABNet'
checkpoint='./checkpoints/'
datasetmode="shadowgttest"
dataroot='' # Need to be specify before testing
name=''  # Need to be specify before testing
resroot='' # Need to be specify before testing
NAME="${name}"

CMD="python ../test.py --loadSize ${loadSize} \
    --name ${NAME} \
    --dataroot  ${dataroot}\
    --checkpoints_dir ${checkpoint} \
    --resroot ${resroot} \
    --size_w   $size_w   --size_h   $size_h\
    --down_w $down_w     --down_h $down_h \
    --fineSize $fineSize --model $model\
    --batch_size $batchs --keep_ratio --phase test_  --gpu_ids ${GPU} \
    --dataset_mode $datasetmode --epoch best \
    --netG ${G}
    $OTHER
"
echo $CMD
eval $CMD

