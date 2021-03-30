import numpy as np
import os
import argparse
import pandas as pd
import pdb
import av
import pdb

"""
    Filter out videos with zero temporal duration or no video stream but audio stream only
"""


np.random.seed(3)

def parse_args():
    parser = argparse.ArgumentParser(description='Build file list')    

    parser.add_argument(
        '--src_listfile_path', type=str, help='source listfile fullpath')
    
    parser.add_argument(
        '--src_videos_root', type=str, help='source video file root dir path')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    
    df = pd.read_csv(args.src_listfile_path, header=None, sep=' ')
    cur_data = df.values

    invalid_video_list = []
    valid_video_list = []
    for i,row in enumerate(cur_data):
        if i%1000 == 0:
            print('Processing {}/{} videos'.format(i,len(cur_data)))
        vid = row[0]
        cur_video_path = os.path.join(args.src_videos_root, vid)        
        try:
            container = av.open(cur_video_path)
            if len(container.streams.video) == 0:
                invalid_video_list.append(cur_video_path)        
            else:
                valid_video_list.append(row)
        except av.error.InvalidDataError:
            invalid_video_list.append(cur_video_path)        
        
    df_filtered = pd.DataFrame(valid_video_list)
    df_filtered.to_csv(args.src_listfile_path+'.filtered', header=None, index=False, sep=' ')

    print('Filtered video file list is written: {}'.format(args.src_listfile_path+'.filtered'))

if __name__ == '__main__':
    main()
