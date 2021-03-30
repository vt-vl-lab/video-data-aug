#!/bin/bash -l
#PBS -A vllab_07
#PBS -W group_list=newriver
#PBS -l walltime=144:00:00
#PBS -l nodes=1:ppn=16:gpus=2
#PBS -q p100_normal_q
#PBS -o /work/huckleberry/jinchoi/temporary_work/action/mmaction/log/tsn/ucf101/testing_pretrained_tsn_ucf101.log
#PBS -e /work/huckleberry/jinchoi/temporary_work/action/mmaction/log/tsn/ucf101/testing_pretrained_tsn_ucf101.log.e

pwd; hostname; date
echo $CUDA_VISIBLE_DEVICES

module load cuda/10.2.89
source activate mmaction_test

cd ~/src/mmaction2
#./tools/dist_train.sh ./configs/recognition/tsn/tsn_r50_1x1x3_80e_ucf101_rgb.py 4
python tools/train.py ./configs/recognition/tsn/tsn_r50_1x1x3_80e_ucf101_rgb.py --validate --gpus 2