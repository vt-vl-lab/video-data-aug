# #!/usr/bin/env bash

# cd ../
# python build_rawframes.py ../../data/kinetics400/videos_train/ ../../data/kinetics400/rawframes_train/ --level 2  --ext mp4 --task rgb --new-width 340 --new-height 256 --use-opencv
# echo "Raw frames (RGB only) generated for train set"

# python build_rawframes.py ../../data/kinetics400/videos_val/ ../../data/kinetics400/rawframes_val/ --level 2 --ext mp4 --task rgb --new-width 340 --new-height 256 --use-opencv
# echo "Raw frames (RGB only) generated for val set"

# cd kinetics400/


#!/usr/bin/env bash

cd ../
python build_rawframes.py ../../data/kinetics400/videos_train/ /home/jinchoi/src/rehab/dataset/action/kinetics/frames_mmaction/train/ --level 2  --ext mp4 --task rgb --new-width 340 --new-height 256 --use-opencv --ref_listfile_path /home/jinchoi/src/rehab/dataset/action/VideoSSL/datalist_from_longlong_jing/kinetics/ssl_sub100c_100p_labeled0.lst
echo "Raw frames (RGB only) generated for train set"

python build_rawframes.py ../../data/kinetics400/videos_val/ /home/jinchoi/src/rehab/dataset/action/kinetics/frames_mmaction/val/ --level 2 --ext mp4 --task rgb --new-width 340 --new-height 256 --use-opencv --ref_listfile_path /home/jinchoi/src/rehab/dataset/action/VideoSSL/datalist_from_longlong_jing/kinetics/ssl_sub100c_100p_val0.lst
echo "Raw frames (RGB only) generated for val set"



# python build_rawframes.py ../../data/kinetics400/videos_val/ /home/jinchoi/src/rehab/dataset/action/kinetics/frames_mmaction_dbg/val/ --level 2 --ext mp4 --task rgb --new-width 340 --new-height 256 --use-opencv --ref_listfile_path /home/jinchoi/src/rehab/dataset/action/VideoSSL/datalist_from_longlong_jing/kinetics/ssl_sub100c_100p_val0.lst

cd kinetics400/
