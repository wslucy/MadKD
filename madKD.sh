#!/bin/bash
#SBATCH -N 1 
#SBATCH -n 5
#SBATCH -M swarm
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
export MODULEPATH=/dat01/paraai_test/software/modulefiles:$MODULEPATH
module load nvidia/cuda/11.6
source activate KD
export PYTHONUNBUFFERED=1

model_t="resnet56"
model_s="resnet20"

# # extra exp
# model_t="resnet110"
# model_s="resnet20"

# model_t="resnet110"
# model_s="resnet32"

# model_t="resnet32x4"
# model_s="resnet8x4"

# model_t="wrn_40_2"
# model_s="wrn_16_2"

# model_t="wrn_40_2"
# model_s="wrn_40_1"

# model_t="vgg13"
# model_s="vgg8"

# model_t="resnet32x4"
# model_s="ShuffleV1"

# model_t="wrn_40_2"
# model_s="ShuffleV1"

# model_t="vgg13"
# model_s="MobileNetV2"

# model_t="resnet50"
# model_s="MobileNetV2"

# model_t="resnet32x4"
# model_s="ShuffleV2"

file_name="out/classification_cifar100/${model_t}_${model_s}"
mkdir -p ${file_name}
for trial in {0..0}; do
    c="1"
    b="2"
    hs="512"
    r="1"
    fw="2"
    rw="1"
    method="madkd"
    path_t="./save/teachers/models/${model_t}_vanilla/ckpt_epoch_240.pth"
    output_file="${file_name}/${method}_hs_${hs}_r_${r}_fw_${fw}_rw_${rw}_relu_c_${c}_b_${b}_${trial}.txt"
    python train_student.py --distill ${method} -hs ${hs} -r ${r} -fw ${fw} -rw ${rw} -c ${c} -d 0 -b ${b} --trial ${trial} --gpu_id ${trial} --path_t ${path_t} --model_s ${model_s} > ${output_file} 2>&1 &
done

wait

python train_student.py --distill madkd -hs 256 -r 1 -fw 1 -rw 1 -c 1 -d 0 -b 2 --trial 1 --gpu_id 0 --path_t ./save/teachers/models/resnet56_vanilla/ckpt_epoch_240.pth --model_s resnet20

# # scripts for TA
# # model_t="vgg13"
# # model_s="wrn_16_2"

# # model_t="wrn_16_4"
# # model_s="wrn_16_2"

# # model_t="resnet50"
# # model_s="wrn_16_2"

# # no T weights
# # model_t="wrn_28_2"
# # model_s="wrn_16_2"

# # trained already
# # model_t="wrn_40_2"
# # model_s="wrn_16_2"

# # model_t="wrn_16_4"
# # model_s="wrn_16_2"

# # no T weights
# # model_t="wrn_28_4"
# # model_s="wrn_16_2"

# file_name="out/classification_cifar100_TA/${model_t}_${model_s}"
# mkdir -p ${file_name}
# for trial in {0..3}; do
#     c="8"
#     b="8"
#     path_t="./save/teachers/models/${model_t}_vanilla/ckpt_epoch_240.pth"
#     output_file="${file_name}/${trial}_c_${c}_b_${b}.txt"
#     python train_student.py --distill cskd -c ${c} -d 0 -b ${b} --trial ${trial} --gpu_id ${trial} --path_t ${path_t} --model_s ${model_s} > ${output_file} 2>&1 &
# done

# wait