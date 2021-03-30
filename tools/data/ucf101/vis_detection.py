import numpy as np
from skvideo.io import ffprobe, vreader, vwrite
import os
import pdb
import glob as gb
import cv2

split = 'val'

src_det_file_path = '/home/jinchoi/src/rehab/dataset/action/kinetics/detectron_results/{}/full_kinetics_detection_{}_rearranged_org_spatial_dim.npy'.format(split,split)
# src_det_file_path = '/home/jinchoi/src/rehab/dataset/action/kinetics/detectron_results/{}/full_kinetics_detection_{}_rearranged.npy'.format(split,split)

videos_root = '/home/jinchoi/src/rehab/dataset/action/kinetics/videos/{}'.format(split)

height_in_tgt_dets = 256 

# read the original detection file
dets = np.load(src_det_file_path, allow_pickle=True).item()

pdb.set_trace()

for i,(cur_cls,vid_datas) in enumerate(dets['dets'].items()):
    cur_cls = cur_cls.replace(' ', '_')

    for k,v in vid_datas.items():
        # get the width and height of the video
        filelist = gb.glob(os.path.join(videos_root, cur_cls, k)+'*')
        if len(filelist) > 0:
            vidfile_name = filelist[0].split('/')[-1]
        input_video_path = os.path.join(videos_root, cur_cls, vidfile_name)
        videogen = vreader(input_video_path)

        vis_vid = []
        resize_dim = 

        for idx,frame in enumerate(videogen):
            cv2.resize(frame, )  
            pdb.set_trace()
            print(' ')