# model settings
model = dict(
    type='SemiBothAugRecognizer3D',
    backbone=dict(
        type='ResNet2Plus1d',
        depth=34,
        pretrained=None,
        pretrained2d=False,
        norm_eval=False,
        conv_cfg=dict(type='Conv2plus1d'),
        norm_cfg=dict(type='SyncBN', requires_grad=True, eps=0.001),
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
        num_classes=51,
        in_channels=512,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01))
# model training and testing settings
train_cfg = dict(actor_cutmix=True)
test_cfg = dict(average_clips=None)
# dataset settings
dataset_type_labeled = 'UnlabeledRawframeDataset'
dataset_type_unlabeled = 'UnlabeledRawframeDataset'
dataset_type_val = 'RawframeDataset'
data_root_labeled = 'data/hmdb51/rawframes/'
data_root_unlabeled = 'data/kinetics400/rawframes_train'
data_root_val = 'data/hmdb51/rawframes/'
split = 1
ann_file_train_labeled = 'data/hmdb51/videossl_splits/hmdb51_train_100_percent_labeled_split_1_rawframes.txt'
ann_file_train_unlabeled = 'data/kinetics400/videossl_splits/kinetics100_train_100_percent_labeled_rawframes.txt'
ann_file_val = 'data/hmdb51/hmdb51_val_split_1_rawframes.txt'
ann_file_test = 'data/hmdb51/hmdb51_val_split_1_rawframes.txt'
# Human box detections
det_file_labeled = 'data/hmdb51/detections.npy'
det_file_unlabeled = f'data/kinetics400/detections.npy'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1),
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
    dict(type='DetectionLoad', thres=0.4),
    dict(type='ResizeWithBox', scale=(-1, 256)),
    dict(type='RandomResizedCropWithBox'),
    dict(type='FlipWithBox', flip_ratio=0.5),
    dict(type='ResizeWithBox', scale=(224, 224), keep_ratio=False),
    dict(type='BuildHumanMask')
]
# Only used for strong augmentation
train_pipeline_strong = [
    dict(type='RandAugment', n=2, m=10),
    dict(type='TemporalAugment')
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
    videos_per_gpu=5,
    workers_per_gpu=0,
    train_labeled=dict(
        type=dataset_type_labeled,
        ann_file=ann_file_train_labeled,
        det_file=det_file_labeled,
        data_prefix=data_root_labeled,
        pipeline_weak=train_pipeline_weak,
        pipeline_strong=train_pipeline_strong,
        pipeline_format=train_pipeline_format),
    train_unlabeled=dict(
        type=dataset_type_unlabeled,
        ann_file=ann_file_train_unlabeled,
        det_file=det_file_unlabeled,
        data_prefix=data_root_unlabeled,
        pipeline_weak=train_pipeline_weak,
        pipeline_strong=train_pipeline_strong,
        pipeline_format=train_pipeline_format),
    val=dict(
        type=dataset_type_val,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type_val,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
        test_mode=True))
# optimizer
optimizer = dict(
    type='SGD', lr=0.2, momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0)
total_epochs = 30
checkpoint_config = dict(interval=5)
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
log_config = dict(
    interval=20, 
    hooks=[
        dict(type='TextLoggerHook')
    ])

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/r2plus1d_r34_8x8x1_30e_hmdb51-kinetics100_rgb_all_full_vidssl_both_aug/'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = False
