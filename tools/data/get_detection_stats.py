import numpy as np
import pandas as pd

import pdb

# detection_file_path = '/home/jinchoi/src/rehab/dataset/action/UCF101/detectron_results_mmaction/ucf101_detections_height_256pixels.npy'

# detection_file_path = '/home/jinchoi/src/rehab/dataset/action/HMDB51/detectron_results/hmdb51_detections_height_256pixels.npy'

detection_file_path = '/home/jinchoi/src/rehab/dataset/action/kinetics/detectron_results/kinetics100/kinetics100_train_detections_height_256pixels.npy'
# detection_file_path = '/home/jinchoi/src/rehab/dataset/action/kinetics/detectron_results/kinetics100/kinetics100_val_detections_height_256pixels.npy'


dets = np.load(detection_file_path, allow_pickle=True).item()
alldets=dets['dets']
conf_threshold = 0.1

all_vid_frms = []
det_indicator = []
for cur_cls, v in alldets.items():
    print('Processing {}'.format(cur_cls))
    for vid, vv in v.items():
        for idx, vvv in vv.items():
            # num_dets = vvv['human_boxes'].shape[0]
            num_dets = vvv['human_boxes'][vvv['human_boxes'][:,-1] >= conf_threshold].shape[0]
            
            if num_dets > 0:
                det_indicator.append(1)
            else:
                det_indicator.append(0)
            cur_vid_frms = '_'.join([cur_cls, vid, 'frame_', '{:05d}'.format(idx)])
            all_vid_frms.append(cur_vid_frms)
            
print('# frames with at least one human detection: {}'.format(sum(det_indicator)))
print('# total frames: {}'.format(len(det_indicator)))
print('The ratio (%) of # frames with at least one human detection and # of total frames: {:0.3f}'.format(100.0*float(sum(det_indicator))/float(len(det_indicator))))
