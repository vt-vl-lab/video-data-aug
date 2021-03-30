#!/bin/bash -l
source activate pt_0.4_p3.6
cd ~/src/video-data-aug

# val set shard 1 of 1
python tools/data/kinetics400/video_jpg_kinetics.py ~/src/rehab/dataset/action/kinetics/videos/val/ ~/src/rehab/dataset/action/kinetics/frames_mmaction_ffmpeg/val ~/src/rehab/dataset/action/VideoSSL/datalist_from_longlong_jing/kinetics/ssl_sub100c_100p_labeled0.lst 0 100

# # train set shard 1 of 5
# python tools/data/kinetics400/video_jpg_kinetics.py ~/src/rehab/dataset/action/kinetics/videos/train/ ~/src/rehab/dataset/action/kinetics/frames_mmaction_ffmpeg/train ~/src/rehab/dataset/action/VideoSSL/datalist_from_longlong_jing/kinetics/ssl_sub100c_100p_labeled0.lst 0 20

# # train set shard 2 of 5
# python tools/data/kinetics400/video_jpg_kinetics.py ~/src/rehab/dataset/action/kinetics/videos/train/ ~/src/rehab/dataset/action/kinetics/frames_mmaction_ffmpeg/train ~/src/rehab/dataset/action/VideoSSL/datalist_from_longlong_jing/kinetics/ssl_sub100c_100p_labeled0.lst 20 40

# # # train set shard 3 of 5
# python tools/data/kinetics400/video_jpg_kinetics.py ~/src/rehab/dataset/action/kinetics/videos/train/ ~/src/rehab/dataset/action/kinetics/frames_mmaction_ffmpeg/train ~/src/rehab/dataset/action/VideoSSL/datalist_from_longlong_jing/kinetics/ssl_sub100c_100p_labeled0.lst 40 60

# # # train set shard 4 of 5
# python tools/data/kinetics400/video_jpg_kinetics.py ~/src/rehab/dataset/action/kinetics/videos/train/ ~/src/rehab/dataset/action/kinetics/frames_mmaction_ffmpeg/train ~/src/rehab/dataset/action/VideoSSL/datalist_from_longlong_jing/kinetics/ssl_sub100c_100p_labeled0.lst 60 80

# # # train set shard 5 of 5
# python tools/data/kinetics400/video_jpg_kinetics.py ~/src/rehab/dataset/action/kinetics/videos/train/ ~/src/rehab/dataset/action/kinetics/frames_mmaction_ffmpeg/train ~/src/rehab/dataset/action/VideoSSL/datalist_from_longlong_jing/kinetics/ssl_sub100c_100p_labeled0.lst 80 100