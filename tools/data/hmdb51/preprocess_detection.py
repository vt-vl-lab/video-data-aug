import numpy as np
import os
import pdb
from PIL import Image, ImageDraw
import cv2
import glob as gb
# import av
# from skvideo.io import ffprobe
import pandas as pd
import sys
import pdb

"""
    convert hmdb-51 detection pickle file to kinetics detection format
"""

src_det_file_path = '/home/jinchoi/src/rehab/dataset/action/HMDB51/detectron_results/detection_merged_missing_dets.pkl'
tgt_det_file_path = '/home/jinchoi/src/rehab/dataset/action/HMDB51/detectron_results/hmdb51_detections_height_256pixels.npy'

dets = np.load(src_det_file_path, allow_pickle=True)

new_fmt_dets = dict()

for i,human_boxes in enumerate(dets['all_boxes'][1]):
    parsed_filename = dets['roidb'][i][0].split('/')
    cur_cls = parsed_filename[-3]
    vid = parsed_filename[-2]
    idx = int(parsed_filename[-1].split('.')[0].split('_')[1])
    
    if cur_cls not in new_fmt_dets:
        new_fmt_dets[cur_cls] = dict()
    
    if vid not in new_fmt_dets[cur_cls]:
        new_fmt_dets[cur_cls][vid] = dict()
    
    new_fmt_dets[cur_cls][vid][int(idx)] = dict(
                                            human_boxes = human_boxes,
                                            frame = str(idx)
                                               )

wrapped_dets = dict(
                     video_res_info=(-1, 256),
                     dets = new_fmt_dets
)

np.save(tgt_det_file_path, wrapped_dets)