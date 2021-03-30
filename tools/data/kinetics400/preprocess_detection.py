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
    convert the detection coordinates (height is always 240) -> (height is always 256)
"""

split = 'val'

src_det_file_path = '/home/jinchoi/src/rehab/dataset/action/kinetics/detectron_results/{}/full_kinetics_detection_{}_rearranged.npy'.format(split,split)

if split == 'val':
    ref_listfile_path = '/home/jinchoi/src/video-data-aug/data/kinetics400/videossl_splits/kinetics100_{}_100_percent_labeled_rawframes.txt'.format(split)
else:
    ref_listfile_path = '/home/jinchoi/src/video-data-aug/data/kinetics400/videossl_splits/kinetics100_{}_100_percent_labeled_rawframes.txt'.format(split)

tgt_det_file_path = '/home/jinchoi/src/rehab/dataset/action/kinetics/detectron_results/{}/kinetics100_{}_detections_height_256pixels.npy'.format(split,split)
frames_root = '/home/jinchoi/src/rehab/dataset/action/kinetics/frames_mmaction_ffmpeg/{}'.format(split)

# read the reference video listfile
df = pd.read_csv(ref_listfile_path, header=None, sep=' ')                
cur_data = df.values
ref_data_dict = {}
ref_cls_list = []
 # contruct a dictionary with class as key and the class/vid, # of frames, label as values
for i,row in enumerate(cur_data):                                
    if i%1000 == 0:
        print('Processing {}/{} videos in the ref. listfile'.format(i+1, cur_data.shape[0]))
    cur_key = row[0].split('/')[0] + '/' + row[0].split('/')[1][:11]
    if cur_key not in ref_data_dict:
        ref_data_dict[cur_key] = row[1:]
    if row[0].split('/')[0] not in ref_cls_list:
        ref_cls_list.append(row[0].split('/')[0])    
    sys.stdout.flush()

print('Done with reading the reference video listfile: {}'.format(ref_listfile_path))

# read the original detection file
# dets = np.load(src_det_file_path, allow_pickle=True).item()
dets = np.load(src_det_file_path)
dets = dets.item()
print('Done with reading the org detection numpy file: {}'.format(src_det_file_path))

height_in_org_dets = 240 # used for SDN project, Hara's 3D-ResNet
height_in_tgt_dets = 256 
scale = float(height_in_tgt_dets)/float(height_in_org_dets)
new_dets = dict()

for i,(cur_cls,vid_datas) in enumerate(dets.items()):
    cur_cls = cur_cls.replace(' ', '_')
    cur_cls = cur_cls.replace('(', '-')
    cur_cls = cur_cls.replace(')', '-')
    print('Processing {}, {}/{}'.format(cur_cls, i+1, len(dets.items())))
    sys.stdout.flush()
    
    for j,(k,v) in enumerate(vid_datas.items()):
        # get the width and height of the video        
        search_key = cur_cls+'/'+k

        if search_key in ref_data_dict:
            print('Processing {}/{} videos in the ref. listfile'.format(j+1, len(vid_datas.items())))
            sys.stdout.flush()
          
            if cur_cls not in new_dets:
                new_dets[cur_cls] = dict()

            for idx, cur_vid_data in v.items():            
                for det in cur_vid_data['human_boxes']:                    
                    det[:4] = det[:4]*scale
            
            new_dets[cur_cls][k] = v

wrapped_dets = dict(
                     video_res_info=(-1, height_in_tgt_dets),
                     dets = new_dets
)        

np.save(tgt_det_file_path, wrapped_dets)
print('Detection results saved to {}'.format(tgt_det_file_path))