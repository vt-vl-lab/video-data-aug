model = dict(
    type='SemiRecognizer3D',
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
train_cfg = dict(actor_cutmix=True)
test_cfg = dict(average_clips=None)
dataset_type = 'RawframeDataset'
dataset_type_unlabeled = 'UnlabeledRawframeDataset'
data_root = 'data/hmdb51/rawframes/'
data_root_val = 'data/hmdb51/rawframes/'
split = 1
ann_file_train_labeled = 'data/hmdb51/videossl_splits/hmdb51_train_40_percent_labeled_split_1_rawframes.txt'
ann_file_train_unlabeled = 'data/hmdb51/videossl_splits/hmdb51_train_100_percent_labeled_split_1_rawframes.txt'
ann_file_val = 'data/hmdb51/hmdb51_val_split_1_rawframes.txt'
ann_file_test = 'data/hmdb51/hmdb51_val_split_1_rawframes.txt'
det_file = 'data/hmdb51/detections.npy'
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
train_pipeline_strong = [
    dict(type='RandAugment', n=2, m=10),
    dict(type='TemporalAugment')
]
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
    workers_per_gpu=2,
    train_labeled=dict(
        type=dataset_type,
        ann_file=ann_file_train_labeled,
        data_prefix=data_root,
        pipeline=train_pipeline),
    train_unlabeled=dict(
        type=dataset_type_unlabeled,
        ann_file=ann_file_train_unlabeled,
        det_file=det_file,
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
optimizer = dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='CosineAnnealing', min_lr=0)
total_epochs = 600
checkpoint_config = dict(interval=10)
evaluation = dict(
    interval=10,
    metrics=['top_k_accuracy', 'mean_class_accuracy'],
    topk=(1, 5))
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/r2plus1d_r34_8x8x1_600e_hmdb51_rgb_all_40percent_vidssl/'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = False
gpu_ids = range(0, 1)
