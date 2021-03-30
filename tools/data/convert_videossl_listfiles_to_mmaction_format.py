import numpy as np
import os
import argparse
import pandas as pd
import glob as gb
import pdb

"""
    Convert VideoSSL listfiles (rawframes/videos) to mmaction listfile format
"""

np.random.seed(3)

# python convert_videossl_listfiles_to_mmaction_format.py ucf101 --src_listfile_dir /home/jinchoi/src/video-data-aug/data/ucf101 --videossl_listfile_dir /home/jinchoi/src/rehab/dataset/action/VideoSSL/datalist/ucf --tgt_listfile_dir /home/jinchoi/src/video-data-aug/data/ucf101/videossl_splits

# python convert_videossl_listfiles_to_mmaction_format.py hmdb51 --src_listfile_dir /home/jinchoi/src/video-data-aug/data/hmdb51 --videossl_listfile_dir /home/jinchoi/src/rehab/dataset/action/VideoSSL/datalist/hmdb --tgt_listfile_dir /home/jinchoi/src/video-data-aug/data/hmdb51/videossl_splits

# python convert_videossl_listfiles_to_mmaction_format.py kinetics100 --src_listfile_dir /home/jinchoi/src/video-data-aug/data/kinetics400 --videossl_listfile_dir /home/jinchoi/src/rehab/dataset/action/VideoSSL/datalist_from_longlong_jing/kinetics --tgt_listfile_dir /home/jinchoi/src/video-data-aug/data/kinetics400/videossl_splits --ext jpg

# python convert_videossl_listfiles_to_mmaction_format.py kinetics100 --src_listfile_dir /home/jinchoi/src/video-data-aug/data/kinetics400 --videossl_listfile_dir /work/vllab/dataset/video_ssl/datalist_from_longlong_jing/kinetics --tgt_listfile_dir /home/jinchoi/src/video-data-aug/data/kinetics400/videossl_splits --ext jpg

def parse_args():
    parser = argparse.ArgumentParser(description='Build file list')    
    parser.add_argument(
    'dataset',
    type=str,
    choices=[
        'ucf101', 'kinetics100', 'hmdb51'
    ],
    help='dataset to be built file list')

    parser.add_argument(
        '--src_listfile_dir', type=str, help='source listfile directory')

    parser.add_argument(
        '--tgt_listfile_dir', type=str, help='target listfile directory')

    parser.add_argument(
        '--videossl_listfile_dir', type=str, help='target listfile directory')

    parser.add_argument(
        '--ext',
        type=str,
        default='mp4',
        choices=['jpg', 'mp4'],
        help='file extensions')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    splits = [1]    
    if args.dataset == 'ucf101':
        ratios = [5, 10, 20, 50]         
        num_classes = '101'
        src_postfix = ''
        phases = ['train']
    elif args.dataset == 'hmdb51':
        # ratios = [5, 10, 20, 40, 50, 60, 100]         
        ratios = [100]         
        num_classes = '51'
        src_postfix = ''
        phases = ['train']
    elif args.dataset == 'kinetics100':
        ratios = [5, 10, 20, 50, 100]         
        num_classes = '100'
        src_postfix = '.filtered'
        phases = ['train', 'val']

    if not os.path.exists(args.tgt_listfile_dir):
        os.makedirs(args.tgt_listfile_dir)

    for split in splits:
        print('Processing {} split {}'.format(args.dataset, split))
        for ratio in ratios:
            for phase in phases:                
                if args.dataset == 'ucf101' or args.dataset == 'hmdb51':
                    src_listfile_path = os.path.join(args.src_listfile_dir, '{}_{}_split_{}_rawframes.txt{}'.format(args.dataset, phase, split, src_postfix))
                    tgt_listfile_path = os.path.join(args.tgt_listfile_dir, '{}_{}_{}_percent_labeled_split_{}_rawframes.txt'.format(args.dataset, phase, ratio, split))
                    # tgt_unlabeled_listfile_path = os.path.join(args.tgt_listfile_dir, '{}_{}_{}_percent_unlabeled_split_{}_rawframes.txt'.format(args.dataset, phase, ratio, split))
                    videossl_listfile_path = os.path.join(args.videossl_listfile_dir, 'ssl_sub{}c_{}p_labeled0.lst'.format(num_classes,ratio))
                else:
                    if args.ext == 'mp4':
                        src_listfile_path = os.path.join(args.src_listfile_dir, 'kinetics400_{}_list_videos.txt{}'.format(phase, src_postfix))
                        tgt_listfile_path = os.path.join(args.tgt_listfile_dir, '{}_{}_{}_percent_labeled_videoss.txt'.format(args.dataset, phase, ratio))
                    elif args.ext == 'jpg':
                        src_listfile_path = os.path.join(args.src_listfile_dir, 'kinetics400_{}_list_rawframes.txt'.format(phase))
                        tgt_listfile_path = os.path.join(args.tgt_listfile_dir, '{}_{}_{}_percent_labeled_rawframes.txt'.format(args.dataset, phase, ratio))
                    if phase == 'val':
                        videossl_listfile_path = os.path.join(args.videossl_listfile_dir, 'ssl_sub100c_100p_val0.lst')                
                    else:
                        videossl_listfile_path = os.path.join(args.videossl_listfile_dir, 'ssl_sub{}c_{}p_labeled0.lst'.format(num_classes,ratio))             
                
                df = pd.read_csv(src_listfile_path, header=None, sep=' ')                
                cur_data = df.values
                cur_data_dict = {}

                # contruct a dictionary with class as key and the class/vid, # of frames, label as values
                for row in cur_data:                                
                    if row[0] not in cur_data_dict:
                        cur_data_dict[row[0]] = row[1:]

                if args.dataset == 'hmdb51' and ratio == 100:
                    videossl_listfile_path = os.path.join(args.videossl_listfile_dir, 'hmdbtrain.lst')
                    df = pd.read_csv(videossl_listfile_path, header=None, sep=' ')                
                    videossl_data = df.values                
                    videossl_listfile_path = os.path.join(args.videossl_listfile_dir, 'hmdbval.lst')
                    df = pd.read_csv(videossl_listfile_path, header=None, sep=' ')                                
                    videossl_data = np.vstack([videossl_data, df.values])
                    pdb.set_trace()
                else:
                    df = pd.read_csv(videossl_listfile_path, header=None, sep='*')                
                    videossl_data = df.values                
                print('{}/{} classes used'.format(np.unique(df[1].values).shape, num_classes))
                
                tgt_data = []
                for row in videossl_data:
                    if args.dataset == 'kinetics100':
                        if args.ext == 'jpg':                            
                            vid = '/'.join(row[0].split('/')[1:]).split('.')[0]                            
                        elif args.ext == 'mp4':
                            vid = '/'.join(row[0].split('/')[1:])
                        else:
                            print('Not implemented for other file types that jpg and mp4')
                        vid = vid.replace(' ', '_').replace('(', '-').replace(')', '-')
                    else:
                        vid = row[0].split('.')[0]                                        

                    if args.dataset == 'hmdb51':
                        if vid not in cur_data_dict:
                            framelist = gb.glob(os.path.join('/home/jinchoi/src/video-data-aug/data/hmdb51/rawframes', vid)+'/*.jpg')                            
                            new_row = [vid] + [len(framelist), row[1]]
                        else:
                            new_row = [vid] + cur_data_dict[vid].tolist()
                        tgt_data.append(new_row)
                    else:
                        if args.dataset == 'kinetics100':
                            if vid in cur_data_dict: 
                                if args.ext == 'mp4':
                                    new_row = [vid] + [str(row[1])]
                                elif args.ext == 'jpg':
                                    new_row = [vid] + [cur_data_dict[vid][0], row[1]]
                                tgt_data.append(new_row)
                        else:
                            new_row = [vid] + cur_data_dict[vid].tolist()
                            tgt_data.append(new_row)
                
                if args.dataset == 'kinetics100':
                    if phase == 'train':
                        np.random.shuffle(tgt_data)     
                    else:
                        tgt_data.sort()

                df_subsampled = pd.DataFrame(tgt_data)
                df_subsampled.to_csv(tgt_listfile_path, header=None, index=False, sep=' ')
                

if __name__ == '__main__':
    main()
