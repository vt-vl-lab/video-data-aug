# model settings
model = dict(
    type='SemiRecognizer3D',
    backbone=dict(
        type='ResNet2Plus1d',
        depth=34,
        pretrained=None,
        pretrained2d=False,
        norm_eval=False,
        conv_cfg=dict(type='Conv2plus1d'),
        norm_cfg=dict(type='SyncBN', requires_grad=True, eps=1e-3),
        act_cfg=dict(type='ReLU'),
        conv1_kernel=(3, 7, 7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        inflate=(1, 1, 1, 1),
        spatial_strides=(1, 2, 2, 2),
        temporal_strides=(1, 2, 2, 2),
        zero_init_residual=False),
    cls_head=dict(
        type='I3DHead',
        num_classes=101,
        in_channels=512,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01))
# model training and testing settings
#train_cfg = None
train_cfg = dict(actor_cutmix=True)
test_cfg = dict(average_clips=None)
# dataset settings
dataset_type = 'RawframeDataset'
dataset_type_unlabeled = 'UnlabeledRawframeDataset'
data_root = 'data/ucf101/rawframes/'
data_root_val = 'data/ucf101/rawframes/'
split = 1  # official train/test splits. valid numbers: 1, 2, 3
ann_file_train_labeled = f'data/ucf101/videossl_splits/ucf101_train_10_percent_labeled_split_{split}_rawframes.txt'
ann_file_train_unlabeled = f'data/ucf101/ucf101_train_split_{split}_rawframes.txt'
ann_file_val = f'data/ucf101/ucf101_val_split_{split}_rawframes.txt'
ann_file_test = f'data/ucf101/ucf101_val_split_{split}_rawframes.txt'
# Human box detections
det_file = f'data/ucf101/detections.npy'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    # dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1),
    # dict(type='DecordDecode'),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
# Get the frame and resize, shared by both weak and strong
train_pipeline_weak = [
    dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='DetectionLoad', thres=0.4),    # Custom imports
    dict(type='ResizeWithBox', scale=(-1, 256)),    # Custom imports
    dict(type='RandomResizedCropWithBox'),    # Custom imports
    dict(type='FlipWithBox', flip_ratio=0.5),    # Custom imports
    dict(type='ResizeWithBox', scale=(224, 224), keep_ratio=False),    # Custom imports
    dict(type='BuildHumanMask'),    # Custom imports
]
# Only used for strong augmentation
train_pipeline_strong = [
    dict(type='RandAugment', n=2, m=10),    # Custom imports
    dict(type='TemporalAugment'),    # Custom imports
]
# Formating the input tensors, shared by both weak and strong
train_pipeline_format = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label', 'human_mask'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label', 'human_mask'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=5,    # NOTE: Need to reduce batch size. 16 -> 5
    workers_per_gpu=4,   # Default: 4
    train_labeled=dict(
        type=dataset_type,
        ann_file=ann_file_train_labeled,
        data_prefix=data_root,
        pipeline=train_pipeline),
    train_unlabeled=dict(
        type=dataset_type_unlabeled,
        ann_file=ann_file_train_unlabeled,
        det_file=det_file,    # Custom imports
        data_prefix=data_root,
        pipeline_weak=train_pipeline_weak,
        pipeline_strong=train_pipeline_strong,
        pipeline_format=train_pipeline_format),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
        test_mode=True))
# optimizer
optimizer = dict(
    type='SGD', lr=0.2, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0)
total_epochs = 360    # Might need to increase this number for different splits. Default: 180
checkpoint_config = dict(interval=5)
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))    # Default: 5
log_config = dict(
    interval=20,    # Default: 20
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/r2plus1d_r34_8x8x1_360e_ucf101_rgb_all_10percent_vidssl/'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = False
