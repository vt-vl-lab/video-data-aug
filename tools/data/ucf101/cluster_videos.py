import os
import argparse
import pdb

"""
Re-organize the UCF-101 dataset frames folders.
src structure: 
├── rawframes
│   ├── v_ApplyEyeMakeup_g01_c01
│   │   ├── img_00001.jpg
│   │   ├── img_00002.jpg
│   │   │   ├── ...
│   ├── v_YoYo_g25_c05
│   │   ├── img_00001.jpg
│   │   ├── img_00002.jpg
│   │   │   ├── ...

tgt: 
├── rawframes
│   ├── ApplyEyeMakeup
│   │   ├── v_ApplyEyeMakeup_g01_c01
│   │   │   ├── img_00001.jpg
│   │   │   ├── img_00002.jpg
│   │   │   │   ├── ...
|   ├── YoYo
│   │   ├── v_YoYo_g25_c05
│   │   │   ├── img_00001.jpg
│   │   │   ├── img_00002.jpg
│   │   │   │   ├── ...
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Build file list')    
    parser.add_argument(
        '--src_folder', type=str, help='root directory for the frames')

    parser.add_argument(
        '--tgt_folder', type=str, help='root directory for the frames to be copied')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    all_dirs = os.listdir(args.src_folder)
    all_dirs.sort()
    action_names = []

    for cur_dir in all_dirs:
        action_name = cur_dir.split('_')[1]
        if action_name not in action_names:
            action_names.append(action_name)
    
    if len(action_names) != 101:
        print('# of action folders are not 101')
        exit()    
    
    if not os.path.exists(args.tgt_folder):
        os.makedirs(args.tgt_folder)

    for i,action_name in enumerate(action_names):
        print('Moving {}/{}: {}'.format(i, len(action_names), action_name))
        os.system('mkdir {}/{}'.format(args.tgt_folder, action_name))        
        cmd = 'mv {}/v_{}* {}/{}'.format(args.src_folder, action_name, args.tgt_folder, action_name)        
        os.system(cmd)

    print('Done re-organizing folders!')

if __name__ == '__main__':
    main()
