# Learning Representational Invariances for Data-Efficient Action Recognition

Official PyTorch implementation for [Learning Representational Invariances for Data-Efficient Action Recognition](). We follow the code structure of [MMAction2](https://github.com/open-mmlab/mmaction2).

See the [project page](https://yuliang.vision/video-data-aug/) for more details.

## Installation

We use PyTorch-1.6.0 with CUDA-10.2 and Torchvision-0.7.0.

Please refer to [install.md](docs/install.md) for installation.


## Data Preparation

First, please download human detection results and put them in the corresponding folder under `data`: [UCF-101](https://filebox.ece.vt.edu/~ylzou/video_data_aug/ucf101/detections.npy), [HMDB-51](https://filebox.ece.vt.edu/~ylzou/video_data_aug/hmdb51/detections.npy), [Kinetics-100](https://filebox.ece.vt.edu/~ylzou/video_data_aug/kinetics400/detections.npy).

Second, please refer to [data_preparation.md](docs/data_preparation.md) to prepare raw frames of UCF-101 and HMDB-51. (Instructions of extracting frames from Kinetics-100 will be available soon.)

(Optional) You can download the pre-extracted ImageNet scores: [UCF-101](https://filebox.ece.vt.edu/~ylzou/video_data_aug/ucf101/resnet18_prtrnd_preds.npy), [HMDB-51](https://filebox.ece.vt.edu/~ylzou/video_data_aug/hmdb51/resnet18_prtrnd_preds.npy), [Kinetics-100](https://filebox.ece.vt.edu/~ylzou/video_data_aug/kinetics400/resnet18_prtrnd_preds_kinetics100_train_subsample_4.npy).


## Training

We use 8 RTX2080 Ti GPUs to run our experiments. You would need to adjust your training schedule accordingly if you have less GPUs. Please refer to [here](docs/getting_started.md#training-setting).

### Supervised learning
```bash
PORT=${PORT:-29500}

python -m torch.distributed.launch \
--nproc_per_node=8 \
--master_port=$PORT \
tools/train.py \
$CONFIG \
--launcher pytorch ${@:3} \
--validate
```

You need to replace `$CONFIG` with the actual config file:
- For supervised baseline, please use config files in `configs/recognition/r2plus1d`.
- For strongly-augmented supervised learning, please use config files in `configs/supervised_aug`.

### Semi-supervised learning
```bash
PORT=${PORT:-29500}

python -m torch.distributed.launch \
--nproc_per_node=8 \
--master_port=$PORT \
tools/train_semi.py \
$CONFIG \
--launcher pytorch ${@:3} \
--validate
```

You need to replace `$CONFIG` with the actual config file:
- For single dataset semi-supervised learning, please use config files in `configs/semi`.
- For cross-dataset semi-supervised learning, please use config files in `configs/semi_both`.

## Testing
```bash
# Multi-GPU testing
./tools/dist_test.sh $CONFIG ${path_to_your_ckpt} ${num_of_gpus} --eval top_k_accuracy

# Single-GPU testing
python tools/test.py $CONFIG ${path_to_your_ckpt} --eval top_k_accuracy
```

**NOTE:** Do not use multi-GPU testing if you are currently using multi-GPU training.

## Other details

Please see [getting_started.md](docs/getting_started.md) for the basic usage of MMAction2.


## Acknowledgement

Codes are built upon [MMAction2](https://github.com/open-mmlab/mmaction2).
