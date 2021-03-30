import argparse
import os

import mmcv
import torch
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.runner.fp16_utils import wrap_fp16_model

from mmaction.apis import multi_gpu_test, single_gpu_test
from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model

import numpy as np
import pdb

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--out', default=None, help='output result file in pickle format')
    parser.add_argument(
        '--pred_result_path_1',
        type=str,
        help='model 1 prediction result file path')
    parser.add_argument(
        '--pred_result_path_2',
        type=str,
        help='model 2 prediction result file path')
    parser.add_argument(
        '--pred_result_path_3',
        type=str,
        help='model 3 prediction result file path')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g.,'
        ' "top_k_accuracy", "mean_class_accuracy" for video dataset')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--options', nargs='+', help='custom options')
    parser.add_argument(
        '--average-clips',
        choices=['score', 'prob'],
        default='score',
        help='average type when averaging test clips')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_configs(cfg1, cfg2):
    # Merge cfg2 into cfg1
    # Overwrite cfg1 if repeated, ignore if value is None.
    cfg1 = {} if cfg1 is None else cfg1.copy()
    cfg2 = {} if cfg2 is None else cfg2
    for k, v in cfg2.items():
        if v:
            cfg1[k] = v
    return cfg1


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)

    # Load output_config from cfg
    output_config = cfg.get('output_config', {})
    # Overwrite output_config from args.out
    output_config = merge_configs(output_config, dict(out=args.out))

    # Load eval_config from cfg
    eval_config = cfg.get('eval_config', {})
    # Overwrite eval_config from args.eval
    eval_config = merge_configs(eval_config, dict(metrics=args.eval))
    # Add options from args.option
    eval_config = merge_configs(eval_config, args.options)

    assert output_config or eval_config, \
        ('Please specify at least one operation (save or eval the '
         'results) with the argument "--out" or "--eval"')

    # set cudnn benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    if cfg.test_cfg is None:
        cfg.test_cfg = dict(average_clips=args.average_clips)
    else:
        cfg.test_cfg.average_clips = args.average_clips

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    data_loader = build_dataloader(
        dataset,
        videos_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
    
    if args.pred_result_path_1:
        model1_preds = np.load(args.pred_result_path_1, allow_pickle=True)
        print('Evaluating {} accuracy:'.format(args.pred_result_path_1))
        eval_res = dataset.evaluate(model1_preds, **eval_config)
    if args.pred_result_path_2:
        model2_preds = np.load(args.pred_result_path_2, allow_pickle=True)
        print('Evaluating {} accuracy:'.format(args.pred_result_path_2))
        eval_res = dataset.evaluate(model2_preds, **eval_config)
    if args.pred_result_path_3:
        model3_preds = np.load(args.pred_result_path_3, allow_pickle=True)
        print('Evaluating {} accuracy:'.format(args.pred_result_path_3))
        eval_res = dataset.evaluate(model3_preds, **eval_config)
    
    if args.pred_result_path_3:
        all_preds = np.stack([model1_preds, model2_preds, model3_preds])
    else:
        all_preds = np.stack([model1_preds, model2_preds])
    ensemble_preds = np.mean(all_preds, axis=0).tolist()

    rank, _ = get_dist_info()
    if rank == 0:
        if output_config:
            out = output_config['out']
            print(f'\nwriting results to {out}')
            dataset.dump_results(outputs, **output_config)
        if eval_config:
            print('Evaluating an ensemble accuracy:')
            eval_res = dataset.evaluate(ensemble_preds, **eval_config)
            for name, val in eval_res.items():
                if 'confusion' not in name: 
                    print(f'{name}: {val:.04f}')
                elif output_config:                    
                    if 'fig' in name:
                        confmat_dir = os.path.dirname(output_config['out'])
                        val.savefig(os.path.join(confmat_dir, name+'.jpg'), format='jpg')

if __name__ == '__main__':
    main()
