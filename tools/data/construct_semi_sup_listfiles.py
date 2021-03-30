import numpy as np
import os
import argparse
import pandas as pd
import pdb

"""
    Constructing custom semi-supervised split for UCF101, HMDB51
"""


np.random.seed(3)

def parse_args():
    parser = argparse.ArgumentParser(description='Build file list')    
    parser.add_argument(
    'dataset',
    type=str,
    choices=[
        'ucf101', 'kinetics400', 'thumos14', 'sthv1', 'sthv2', 'mit',
        'mmit', 'activitynet', 'hmdb51'
    ],
    help='dataset to be built file list')

    parser.add_argument(
        '--src_listfile_dir', type=str, help='source listfile directory')

    parser.add_argument(
        '--tgt_listfile_dir', type=str, help='target listfile directory')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    splits = [1,2,3]
    phases = ['train']
    if args.dataset == 'ucf101':
        ratios = [1, 2, 5, 10, 20, 50]         
    elif args.dataset == 'hmdb51':
        ratios = [5, 10, 20, 30, 50, 60]         

    if not os.path.exists(args.tgt_listfile_dir):
        os.makedirs(args.tgt_listfile_dir)

    for split in splits:
        print('Processing {} split {}'.format(args.dataset, split))
        for ratio in ratios:
            for phase in phases:                
                src_listfile_path = os.path.join(args.src_listfile_dir, '{}_{}_split_{}_rawframes.txt'.format(args.dataset, phase, split))
                tgt_listfile_path = os.path.join(args.tgt_listfile_dir, '{}_{}_{}_percent_labeled_split_{}_rawframes.txt'.format(args.dataset, phase, ratio, split))
                tgt_unlabeled_listfile_path = os.path.join(args.tgt_listfile_dir, '{}_{}_{}_percent_unlabeled_split_{}_rawframes.txt'.format(args.dataset, phase, ratio, split))
                # pdb.set_trace()
                df = pd.read_csv(src_listfile_path, header=None, sep=' ')
                # cur_data = df.to_numpy()                
                cur_data = df.values
                cur_data_dict = {}

                # contruct a dictionary with class as key and the class/vid, # of frames, label as values
                for row in cur_data:
                    cur_cls, cur_vid = row[0].split('/')
                    if cur_cls not in cur_data_dict:
                        cur_data_dict[cur_cls] = []
                    cur_data_dict[cur_cls].append(row)
                
                # subsample videos for each class                
                subsampled_data_dict = dict()
                subsampled_data_dict_unlabeled = dict()
                for key,val in cur_data_dict.items():
                    num_cur_class_examples = len(val)
                    num_subsampled_cur_class_examples = max(int(np.round(num_cur_class_examples*(ratio/100.0))), 1)
                    sampled_vid_inds = np.random.choice(num_cur_class_examples, num_subsampled_cur_class_examples, replace=False)
                    subsampled_data_dict[key] = np.array(val)[sampled_vid_inds]
                    unlabeled_ind = np.array(list(set(np.arange(num_cur_class_examples))-set(sampled_vid_inds)))
                    subsampled_data_dict_unlabeled[key] = np.array(val)[unlabeled_ind]                    

                # write to the original kinetics csv format
                num_subsampled_vids, num_subsampled_vids_unlabeled = 0,0
                subsampled_csv_np = np.empty((0,3),dtype=object)                
                for key,val in subsampled_data_dict.items():
                    subsampled_csv_np = np.vstack([subsampled_csv_np, val])                    
                    num_subsampled_vids += len(val)
                
                df_subsampled = pd.DataFrame(subsampled_csv_np)
                df_subsampled.to_csv(tgt_listfile_path, header=None, index=False, sep=' ')

                subsampled_unlabeled_csv_np = np.empty((0,2),dtype=object)
                for key,val in subsampled_data_dict_unlabeled.items():            
                    subsampled_unlabeled_csv_np = np.vstack([subsampled_unlabeled_csv_np, val[:, :2]])
                    num_subsampled_vids_unlabeled += len(val)                

                print('Original # of videos: {}, {:.2f}/{} percent subsampled # of labeled videos: {}, {:.2f}/{} percent subsampled # of unlabeled videos: {}'.format(len(cur_data), float(num_subsampled_vids)/float(len(cur_data))*100.0, ratio, num_subsampled_vids, float(num_subsampled_vids_unlabeled)/float(len(cur_data))*100.0, 100-ratio, num_subsampled_vids_unlabeled))

                df_subsampled_unlabeled = pd.DataFrame(subsampled_unlabeled_csv_np)
                df_subsampled_unlabeled.to_csv(tgt_unlabeled_listfile_path, header=None, index=False, sep=' ')
                

if __name__ == '__main__':
    main()
