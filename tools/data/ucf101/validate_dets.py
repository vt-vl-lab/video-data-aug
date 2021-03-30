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

""""
    Check the # of frames in the detection cache file and the # of frames in the extracted frames folder match or not
"""

src_det_file_path = '/home/jinchoi/src/rehab/dataset/action/UCF101/detectron_results_mmaction/ucf101_detections_height_256pixels.npy'

frames_root = '/home/jinchoi/src/rehab/dataset/action/UCF101/UCF101_frames_mmaction'
tgt_diff_num_frms_path = '/home/jinchoi/src/rehab/dataset/action/UCF101/detectron_results_mmaction/diff_num_frms_mmaction_ffmpeg_detections.csv'

dets = np.load(src_det_file_path)
dets = dets.item()
dets = dets['dets']
print('Done with reading the org detection numpy file: {}'.format(src_det_file_path))

frm_num_not_matched = dict()
frm_nums = []
keys = []

for i,(cur_cls,vid_datas) in enumerate(dets.items()):
    print('Processing {}, {}/{}'.format(cur_cls, i+1, len(dets.items())))
    sys.stdout.flush()
    if cur_cls == 'HandStandPushups':
        cur_cls = 'HandstandPushups'

    for j,(k,v) in enumerate(vid_datas.items()):
        # get the width and height of the video        
        
        search_key = cur_cls+'/'+k

        print('Processing {}/{} videos in the ref. listfile'.format(j+1, len(vid_datas.items())))
        sys.stdout.flush()

        cur_dir = os.path.join(frames_root, cur_cls, k)
        filelist = os.listdir(cur_dir)
        org_n_frames = len(filelist)

        if org_n_frames != len(v.items()):
            print('Video file length is different from the data in the detection data: {}/{}'.format(org_n_frames, len(v.items())))
            frm_num_not_matched[search_key] = [org_n_frames, len(v.items())]
            frm_nums.append([org_n_frames,len(v.items())])
            keys.append(search_key)

if len(frm_num_not_matched) == 0:
    print('# of frames matched!!!')
else:
    print('There are some mismatch between the # frames in extracted frames: {} and # frames in cached detections: {}'.format(frames_root, src_det_file_path))

    frm_nums = np.array(frm_nums)
    keys = np.array(keys)

    if np.sum(frm_nums[:,0]-frm_nums[:,1]>=0) == 0:
        print('mmaction # of frames are always less than or equal to the # of frames used for detection')
    else:
        print('mmaction # of frames are NOT always less than or equal to the # of frames used for detection')

    df = pd.DataFrame(data={'vid': keys, 'mmaction # frms':frm_nums[:,0], 'ffmpeg # frms': frm_nums[:,1], 'diff': frm_nums[:,0]-frm_nums[:,1]})
    df.to_csv(tgt_diff_num_frms_path, index=None, sep=' ')

    pdb.set_trace()
    print('')
