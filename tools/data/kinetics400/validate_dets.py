import numpy as np
import os
import pdb
from PIL import Image, ImageDraw
import glob as gb
# import av
import pandas as pd
import sys

""""
    Check the # of frames in the detection cache file and the # of frames in the extracted frames folder match or not
"""
split = 'val'

# src_det_file_path = '/home/jinchoi/src/rehab/dataset/action/kinetics/detectron_results/{}/full_kinetics_detection_{}_rearranged.npy'.format(split,split)
src_det_file_path = '/home/jinchoi/src/rehab/dataset/action/kinetics/detectron_results/kinetics100/kinetics100_{}_detections_height_256pixels.npy'.format(split)

frames_root = '/home/jinchoi/src/rehab/dataset/action/kinetics/frames_mmaction_ffmpeg/{}'.format(split)
tgt_diff_num_frms_path = '/home/jinchoi/src/rehab/dataset/action/kinetics/detectron_results/diff_num_frms_mmaction_ffmpeg_detections_{}.csv'.format(split)

dets = np.load(src_det_file_path, allow_pickle=True)
dets = dets.item()
dets = dets['dets']
print('Done with reading the org detection numpy file: {}'.format(src_det_file_path))

frm_num_not_matched = dict()
frm_nums = []
keys = []

for i,(cur_cls,vid_datas) in enumerate(dets.items()):
    print('Processing {}, {}/{}'.format(cur_cls, i+1, len(dets.items())))
    sys.stdout.flush()
    cur_cls = cur_cls.replace(' ', '_').replace('(', '-').replace(')', '-')
    
    for j,(k,v) in enumerate(vid_datas.items()):
        # get the width and height of the video        
        if os.path.exists(os.path.join(frames_root, cur_cls)):
            search_key = cur_cls+'/'+k
            print('Processing {}/{} videos in the ref. listfile'.format(j+1, len(vid_datas.items())))
            sys.stdout.flush()            

            cur_dir = gb.glob(os.path.join(frames_root, cur_cls, k)+'*')[0]

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
