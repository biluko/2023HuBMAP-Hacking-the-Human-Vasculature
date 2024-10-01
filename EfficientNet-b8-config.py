model = dict(
    type='CascadeRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    backbone=dict(
        type='EfficientNet',
        arch='b8',
        drop_path_rate=0.2,
        out_indices=(
            2,
            3,
            4,
            5,
        ),
        frozen_stages=0,
        norm_cfg=dict(
            type='SyncBN', requires_grad=True, eps=0.001, momentum=0.01),
        norm_eval=False,
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone',
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b8_3rdparty_8xb32-aa-advprop_in1k_20220119-297ce1b7.pth'
        )),
    neck=dict(
        type='FPN',
        in_channels=[
            56,
            88,
            248,
            704,
        ],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[
                8,
            ],
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[
            1,
            0.5,
            0.25,
        ],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[
                4,
                8,
                16,
                32,
            ]),
        bbox_head=[
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=2048,
                roi_feat_size=7,
                num_classes=3,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.1,
                        0.1,
                        0.2,
                        0.2,
                    ]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=2048,
                roi_feat_size=7,
                num_classes=3,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.05,
                        0.05,
                        0.1,
                        0.1,
                    ]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=2048,
                roi_feat_size=7,
                num_classes=3,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.033,
                        0.033,
                        0.067,
                        0.067,
                    ]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
        ],
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[
                4,
                8,
                16,
                32,
            ]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=3,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.6),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.6),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.01,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))
dataset_type = 'CocoDataset'
data_root = 'D:/CodeProject/HuBMAP2023/Data/'
backend_args = None
albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[
            0.1,
            0.3,
        ],
        contrast_limit=[
            0.1,
            0.3,
        ],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0),
        ],
        p=0.1),
    dict(type='ImageCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0),
        ],
        p=0.1),
]
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Albu',
        transforms=[
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[
                    0.1,
                    0.3,
                ],
                contrast_limit=[
                    0.1,
                    0.3,
                ],
                p=0.2),
            dict(
                type='OneOf',
                transforms=[
                    dict(
                        type='RGBShift',
                        r_shift_limit=10,
                        g_shift_limit=10,
                        b_shift_limit=10,
                        p=1.0),
                    dict(
                        type='HueSaturationValue',
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=1.0),
                ],
                p=0.1),
            dict(
                type='ImageCompression',
                quality_lower=85,
                quality_upper=95,
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0),
                ],
                p=0.1),
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap=dict(img='image', gt_masks='masks', gt_bboxes='bboxes'),
        skip_img_without_anno=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[
                        (
                            512,
                            512,
                        ),
                        (
                            640,
                            640,
                        ),
                        (
                            720,
                            720,
                        ),
                        (
                            1024,
                            1024,
                        ),
                        (
                            1080,
                            1080,
                        ),
                    ],
                    keep_ratio=True),
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[
                        (
                            1024,
                            1024,
                        ),
                        (
                            1080,
                            1080,
                        ),
                        (
                            840,
                            840,
                        ),
                    ],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(
                        1024,
                        1024,
                    ),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[
                        (
                            512,
                            512,
                        ),
                        (
                            640,
                            640,
                        ),
                        (
                            720,
                            720,
                        ),
                        (
                            1024,
                            1024,
                        ),
                        (
                            1080,
                            1080,
                        ),
                    ],
                    keep_ratio=True),
            ],
        ]),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(
        1024,
        1024,
    ), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        )),
]
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoDataset',
        data_root='D:/CodeProject/HuBMAP2023/Data/',
        ann_file='D:/CodeProject/HuBMAP2023/Data/coco_annotations_train_all.json',
        data_prefix=dict(img='train_data/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                type='Albu',
                transforms=[
                    dict(
                        type='ShiftScaleRotate',
                        shift_limit=0.0625,
                        scale_limit=0.0,
                        rotate_limit=0,
                        interpolation=1,
                        p=0.5),
                    dict(
                        type='RandomBrightnessContrast',
                        brightness_limit=[
                            0.1,
                            0.3,
                        ],
                        contrast_limit=[
                            0.1,
                            0.3,
                        ],
                        p=0.2),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='RGBShift',
                                r_shift_limit=10,
                                g_shift_limit=10,
                                b_shift_limit=10,
                                p=1.0),
                            dict(
                                type='HueSaturationValue',
                                hue_shift_limit=20,
                                sat_shift_limit=30,
                                val_shift_limit=20,
                                p=1.0),
                        ],
                        p=0.1),
                    dict(
                        type='ImageCompression',
                        quality_lower=85,
                        quality_upper=95,
                        p=0.2),
                    dict(type='ChannelShuffle', p=0.1),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='Blur', blur_limit=3, p=1.0),
                            dict(type='MedianBlur', blur_limit=3, p=1.0),
                        ],
                        p=0.1),
                ],
                bbox_params=dict(
                    type='BboxParams',
                    format='pascal_voc',
                    label_fields=[
                        'gt_bboxes_labels',
                        'gt_ignore_flags',
                    ],
                    min_visibility=0.0,
                    filter_lost_elements=True),
                keymap=dict(img='image', gt_masks='masks', gt_bboxes='bboxes'),
                skip_img_without_anno=True),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='RandomChoice',
                transforms=[
                    [
                        dict(
                            type='RandomChoiceResize',
                            scales=[
                                (
                                    512,
                                    512,
                                ),
                                (
                                    640,
                                    640,
                                ),
                                (
                                    720,
                                    720,
                                ),
                                (
                                    1024,
                                    1024,
                                ),
                                (
                                    1080,
                                    1080,
                                ),
                            ],
                            keep_ratio=True),
                    ],
                    [
                        dict(
                            type='RandomChoiceResize',
                            scales=[
                                (
                                    1024,
                                    1024,
                                ),
                                (
                                    1080,
                                    1080,
                                ),
                                (
                                    840,
                                    840,
                                ),
                            ],
                            keep_ratio=True),
                        dict(
                            type='RandomCrop',
                            crop_type='absolute_range',
                            crop_size=(
                                1024,
                                1024,
                            ),
                            allow_negative_crop=True),
                        dict(
                            type='RandomChoiceResize',
                            scales=[
                                (
                                    512,
                                    512,
                                ),
                                (
                                    640,
                                    640,
                                ),
                                (
                                    720,
                                    720,
                                ),
                                (
                                    1024,
                                    1024,
                                ),
                                (
                                    1080,
                                    1080,
                                ),
                            ],
                            keep_ratio=True),
                    ],
                ]),
            dict(type='PackDetInputs'),
        ],
        backend_args=None,
        metainfo=dict(classes=(
            'glomerulus',
            'blood_vessel',
            'unsure',
        ))))
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='D:/CodeProject/HuBMAP2023/Data/',
        ann_file='D:/CodeProject/HuBMAP2023/Data/coco_annotations_valid_all.json',
        data_prefix=dict(img='valid_data/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(
                1024,
                1024,
            ), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                type='PackDetInputs',
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                )),
        ],
        backend_args=None,
        metainfo=dict(classes=(
            'glomerulus',
            'blood_vessel',
            'unsure',
        ))))
test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='D:/CodeProject/HuBMAP2023/Data/',
        ann_file='D:/CodeProject/HuBMAP2023/Data/coco_annotations_valid_all.json',
        data_prefix=dict(img='valid_data/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(
                1024,
                1024,
            ), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                type='PackDetInputs',
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                )),
        ],
        backend_args=None,
        metainfo=dict(classes=(
            'glomerulus',
            'blood_vessel',
            'unsure',
        ))))
val_evaluator = dict(
    type='CocoMetric',
    ann_file='D:/CodeProject/HuBMAP2023/Data/coco_annotations_valid_all.json',
    metric=[
        'segm',
    ],
    format_only=False,
    backend_args=None)
test_evaluator = dict(
    type='CocoMetric',
    ann_file='D:/CodeProject/HuBMAP2023/Data/coco_annotations_valid_all.json',
    metric=[
        'segm',
    ],
    format_only=False,
    backend_args=None)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[
            27,
            33,
        ],
        gamma=0.1),
]
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0002, betas=(
            0.9,
            0.999,
        ), weight_decay=0.01))
auto_scale_lr = dict(enable=False, base_batch_size=16)
default_scope = 'mmdet'
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='coco/segm_precision',
        save_optimizer=False,
        save_param_scheduler=False))
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='coco/segm_precision',
        rule='greater',
        min_delta=0.001,
        strict=False,
        check_finite=True,
        patience=10,
        stopping_threshold=None),
]
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
custom_imports = dict(
    imports=[
        'mmpretrain.models',
    ], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-base_fcmae-in21k-pre_3rdparty_in1k-384px_20230104-379425cc.pth'
max_epochs = 36
launcher = 'none'
work_dir = 'D:/CodeProject/HuBMAP2023/Data/work_dirs/EfficientNet-b8-config'
