#!/usr/bin/env bash

cd ~/src/video-data-aug
source activate video-data-aug

# test time ensemble
# (a,b) means training time cascade of a and b, a+b means test time ensemble of a and b

echo "(RandAug,TempAugAll) + RandAug"
python ./tools/test_ensemble.py ./work_dirs/r2plus1d_r34_video_8x8x1_180e_ucf101_rgb_all_20percent_vidssl/r2plus1d_r34_video_8x8x1_180e_ucf101_rgb_all_20percent_vidssl.py \
--pred_result_path_1 ./work_dirs/r2plus1d_r34_video_8x8x1_180e_ucf101_rgb_rand_temporal_augment_20percent_vidssl_20201027/results.pkl \
--pred_result_path_2 ./work_dirs/r2plus1d_r34_video_8x8x1_180e_ucf101_rgb_fixmatch_20percent_vidssl_20201027/results.pkl \
--eval top_k_accuracy

echo "(RandAug,TempAugAll) + TempAugAll"
python ./tools/test_ensemble.py ./work_dirs/r2plus1d_r34_video_8x8x1_180e_ucf101_rgb_all_20percent_vidssl/r2plus1d_r34_video_8x8x1_180e_ucf101_rgb_all_20percent_vidssl.py \
--pred_result_path_1 ./work_dirs/r2plus1d_r34_video_8x8x1_180e_ucf101_rgb_rand_temporal_augment_20percent_vidssl_20201027/results.pkl \
--pred_result_path_2 ./work_dirs/r2plus1d_r34_video_8x8x1_180e_ucf101_rgb_taugment_20percent_vidssl_20201027/results.pkl \
--eval top_k_accuracy

echo "ActorCutMix_v2 + TempAugAll"
python ./tools/test_ensemble.py ./work_dirs/r2plus1d_r34_video_8x8x1_180e_ucf101_rgb_all_20percent_vidssl/r2plus1d_r34_video_8x8x1_180e_ucf101_rgb_all_20percent_vidssl.py \
--pred_result_path_1 ./work_dirs/r2plus1d_r34_video_8x8x1_180e_ucf101_rgb_actorcutmix_20percent_label_smoothing_vidssl/results.pkl \
--pred_result_path_2 ./work_dirs/r2plus1d_r34_video_8x8x1_180e_ucf101_rgb_taugment_20percent_vidssl_20201027/results.pkl \
--eval top_k_accuracy

echo "ActorCutMix_v2 + (RandAug,TempAugAll)"
python ./tools/test_ensemble.py ./work_dirs/r2plus1d_r34_video_8x8x1_180e_ucf101_rgb_all_20percent_vidssl/r2plus1d_r34_video_8x8x1_180e_ucf101_rgb_all_20percent_vidssl.py \
--pred_result_path_1 ./work_dirs/r2plus1d_r34_video_8x8x1_180e_ucf101_rgb_actorcutmix_20percent_label_smoothing_vidssl/results.pkl \
--pred_result_path_2 ./work_dirs/r2plus1d_r34_video_8x8x1_180e_ucf101_rgb_rand_temporal_augment_20percent_vidssl_20201027/results.pkl \
--eval top_k_accuracy

echo "ActorCutMix_v2 + RandAug"
python ./tools/test_ensemble.py ./work_dirs/r2plus1d_r34_video_8x8x1_180e_ucf101_rgb_all_20percent_vidssl/r2plus1d_r34_video_8x8x1_180e_ucf101_rgb_all_20percent_vidssl.py \
--pred_result_path_1 ./work_dirs/r2plus1d_r34_video_8x8x1_180e_ucf101_rgb_actorcutmix_20percent_label_smoothing_vidssl/results.pkl \
--pred_result_path_2 ./work_dirs/r2plus1d_r34_video_8x8x1_180e_ucf101_rgb_fixmatch_20percent_vidssl_20201027/results.pkl \
--eval top_k_accuracy

echo "TempAugAll + RandAug"
python ./tools/test_ensemble.py ./work_dirs/r2plus1d_r34_video_8x8x1_180e_ucf101_rgb_all_20percent_vidssl/r2plus1d_r34_video_8x8x1_180e_ucf101_rgb_all_20percent_vidssl.py \
--pred_result_path_1 ./work_dirs/r2plus1d_r34_video_8x8x1_180e_ucf101_rgb_taugment_20percent_vidssl_20201027/results.pkl \
--pred_result_path_2 ./work_dirs/r2plus1d_r34_video_8x8x1_180e_ucf101_rgb_rand_temporal_augment_20percent_vidssl_20201027/results.pkl \
--eval top_k_accuracy

echo "ActorCutMix_v2 + RandAug + TempAugAll"
python ./tools/test_ensemble.py ./work_dirs/r2plus1d_r34_video_8x8x1_180e_ucf101_rgb_all_20percent_vidssl/r2plus1d_r34_video_8x8x1_180e_ucf101_rgb_all_20percent_vidssl.py \
--pred_result_path_1 ./work_dirs/r2plus1d_r34_video_8x8x1_180e_ucf101_rgb_actorcutmix_20percent_label_smoothing_vidssl/results.pkl \
--pred_result_path_2 ./work_dirs/r2plus1d_r34_video_8x8x1_180e_ucf101_rgb_fixmatch_20percent_vidssl_20201027/results.pkl \
--pred_result_path_3 ./work_dirs/r2plus1d_r34_video_8x8x1_180e_ucf101_rgb_taugment_20percent_vidssl_20201027/results.pkl \
--eval top_k_accuracy --eval top_k_accuracy