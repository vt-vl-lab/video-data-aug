import numpy as np
import os
import pdb
from PIL import Image, ImageDraw
import cv2
import glob as gb
# import av
from skvideo.io import ffprobe
import pandas as pd
import sys

"""
    convert ucf-101 detection pickle file to kinetics detection format
"""

src_det_file_path = '/home/jinchoi/src/rehab/dataset/action/UCF101/detectron_results_mmaction/detections_merged_rearranged.npy'
tgt_det_file_path = '/home/jinchoi/src/rehab/dataset/action/UCF101/detectron_results_mmaction/ucf101_detections_height_256pixels.npy'

dets = np.load(src_det_file_path, allow_pickle=True).item()

wrapped_dets = dict(
                     video_res_info=(-1, 256),
                     dets = dets
)

np.save(tgt_det_file_path, wrapped_dets)



# src_det_file_path = '/home/jinchoi/src/rehab/dataset/action/UCF101/detectron_results_mmaction/detections_merged_rearranged.npy'
# tgt_det_file_path = '/home/jinchoi/src/rehab/dataset/action/UCF101/detectron_results_mmaction/ucf101_detections_height_256pixels.npy'

# dets = np.load(src_det_file_path, allow_pickle=True)
# # dets = np.load(src_det_file_path, allow_pickle=True).item()

# classes = []
# new_fmt_dets = dict()

# for k,v in dets.items():
#     vid, idx = k.split('/')
#     cur_cls = k.split('_')[1]
    
#     if cur_cls not in new_fmt_dets:
#         new_fmt_dets[cur_cls] = dict()
    
#     if vid not in new_fmt_dets[cur_cls]:
#         new_fmt_dets[cur_cls][vid] = dict()
    
#     new_fmt_dets[cur_cls][vid][int(idx)] = dict(
#                                             human_boxes = v['human_boxes'],
#                                             frame = idx
#                                                )

# wrapped_dets = dict(
#                      video_res_info=(-1, 256),
#                      dets = new_fmt_dets
# )

# np.save(tgt_det_file_path, wrapped_dets)