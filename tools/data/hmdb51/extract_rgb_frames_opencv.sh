#!/usr/bin/env bash

cd ../
# python build_rawframes.py ../../data/hmdb51/videos/ ../../data/hmdb51/rawframes/ --task rgb --level 2 --ext avi --use-opencv
python build_rawframes.py /home/jinchoi/src/rehab/dataset/action/HMDB51/videos /home/jinchoi/src/rehab/dataset/action/HMDB51/HMDB51_frames_mmaction --task rgb --level 2 --ext avi --use-opencv --new-short 256

echo "Genearte raw frames (RGB only)"

cd hmdb51/
