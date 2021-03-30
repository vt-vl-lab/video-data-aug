#!/usr/bin/env bash

cd ../
python build_rawframes.py /home/jinchoi/src/rehab/dataset/action/UCF101/UCF101_videos /home/jinchoi/src/rehab/dataset/action/UCF101/UCF101_frames_mmaction --task rgb --level 2 --ext avi --use-opencv --new-short 256
echo "Genearte raw frames (RGB only)"

cd ucf101/
