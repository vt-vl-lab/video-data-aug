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
    Add keys for videos without any detections, add also frame keys for those videos
"""

split = 'train'

src_det_file_path = '/home/jinchoi/src/rehab/dataset/action/kinetics/detectron_results/kinetics100/kinetics100_{}_detections_height_256pixels.npy'.format(split)

if split == 'val':
    ref_listfile_path = '/home/jinchoi/src/video-data-aug/data/kinetics400/videossl_splits/kinetics100_{}_100_percent_labeled_rawframes.txt'.format(split)
else:
    ref_listfile_path = '/home/jinchoi/src/video-data-aug/data/kinetics400/videossl_splits/kinetics100_{}_100_percent_labeled_rawframes.txt'.format(split)

tgt_det_file_path = '/home/jinchoi/src/rehab/dataset/action/kinetics/detectron_results/kinetics100/kinetics100_{}_no_missing_keys_detections_height_256pixels.npy'.format(split)

# read the original detection file
dets = np.load(src_det_file_path, allow_pickle=True)
dets = dets.item()
video_res_info = dets['video_res_info']
dets = dets['dets']
print('Done with reading the org detection numpy file: {}'.format(src_det_file_path))

# read the reference video listfile
df = pd.read_csv(ref_listfile_path, header=None, sep=' ')                
ref_data = df.values

new_dets = dict()
new_dets = {}
ref_cls_list = []
missing_det_vid_cnt = 0
 # contruct a dictionary with class as key and the class/vid, # of frames, label as values
for i,row in enumerate(ref_data):                                
    if i%1000 == 0:
        print('Processing {}/{} videos in the ref. listfile'.format(i+1, ref_data.shape[0]))
    # cur_key = row[0].split('/')[0] + '/' + row[0].split('/')[1][:11]
    cur_cls = row[0].split('/')[0]
    cur_vid = row[0].split('/')[1][:11]
    num_frms = row[1]
    if cur_cls not in new_dets:
        new_dets[cur_cls] = dict()
    
    if cur_vid not in new_dets[cur_cls]:
        new_dets[cur_cls][cur_vid] = dict()
        for idx in range(num_frms):
            idx_one_based = idx + 1
            new_dets[cur_cls][cur_vid][idx_one_based] = {'frame'
            : idx_one_based, 'human_boxes': np.zeros([0,5]).astype(np.float32)}

    if cur_vid in dets[cur_cls]:
        assert len(new_dets[cur_cls][cur_vid]) == len(dets[cur_cls][cur_vid])
        new_dets[cur_cls][cur_vid] = dets[cur_cls][cur_vid]        
        dets[cur_cls].pop(cur_vid,None)
    else:
        missing_det_vid_cnt += 1
        print(i, cur_vid)
            
    if cur_cls not in ref_cls_list:
        ref_cls_list.append(cur_cls)    
    sys.stdout.flush()

print('Done with adding missing vid keys and frame keys by comparing {} and {}'.format(ref_listfile_path, src_det_file_path))

# validate if all the exisiting dets are copied to the new_dets
for cur_cls,cur_data in dets.items():
    if len(cur_data.keys()) > 0:
        pdb.set_trace()

wrapped_dets = dict(
                     video_res_info=video_res_info,
                     dets = new_dets
)

np.save(tgt_det_file_path, wrapped_dets)
print('Detection results saved to {}'.format(tgt_det_file_path))