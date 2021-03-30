from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16

from .. import builder

# Custom imports
from .base import BaseRecognizer


class SemiBaseRecognizer(BaseRecognizer):
    """Base class for semi-supervised recognizers.

    All recognizers should subclass it.
    All subclass should overwrite:

    - Methods:``forward_train``, supporting to forward when training.
    - Methods:``forward_test``, supporting to forward when testing.

    Args:
        backbone (dict): Backbone modules to extract feature.
        cls_head (dict): Classification head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
    """

    def __init__(self, backbone, cls_head, train_cfg=None, test_cfg=None):
        super().__init__(backbone, cls_head, train_cfg=train_cfg, test_cfg=test_cfg)
        if train_cfg:
            self.actor_cutmix = train_cfg.get('actor_cutmix', False)
            self.knowledge_distill = train_cfg.get('knowledge_distill', False)
            self.framework = train_cfg.get('framework', 'fixmatch')
        else:
            self.actor_cutmix = False
            self.knowledge_distill = False
            self.framework = 'fixmatch'

        # Build a distillation cls head
        if self.knowledge_distill:
            distill_head = cls_head.copy()
            distill_head['num_classes'] = 1000
            self.distill_head = builder.build_head(distill_head)
            self.distill_head.init_weights()

    @abstractmethod
    def forward_train(self, imgs, labels, imgs_weak, imgs_strong, labels_unlabeled):
        """Defines the computation performed at every call when training."""
        pass

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        if dist.is_available() and dist.is_initialized():
            loss_value = log_vars['num_select'].data.clone()
            dist.all_reduce(loss_value)
            log_vars['num_select'] = loss_value.item()

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                if loss_name == 'num_select':
                    continue

                loss_value = loss_value.data.clone()
                # NOTE: Compute unlabeled data accuracy for selected confident examples only
                if 'acc_weak' in loss_name or 'acc_strong' in loss_name:
                    dist.all_reduce(loss_value)
                    loss_value.div_(log_vars['num_select']+1e-6)
                    # TODO: A work-around. Sometimes all 0 accuracy will explode after all_reduce
                    if loss_value > 1. or loss_value < 0.:
                        loss_value = torch.zeros(1).to(loss_value.device)
                else:
                    dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def forward(self, imgs, label=None, return_loss=True, imgs_weak=None, imgs_strong=None, label_unlabeled=None, human_mask=None, imagenet_prob=None):
        """Define the computation performed at every call."""
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            if imgs_weak is None or imgs_strong is None or label_unlabeled is None:
                raise ValueError('Unlabeled data should be available.')
            return self.forward_train(imgs, label, imgs_weak, imgs_strong, label_unlabeled, human_mask, imagenet_prob)
        else:
            return self.forward_test(imgs)

    def train_step(self, data_batch_labeled, data_batch_unlabeled, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch_labeled (dict): The output of dataloader (labeled).
            data_batch_unlabeled (dict): The output of dataloader (unlabeled).
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        imgs = data_batch_labeled['imgs']
        label = data_batch_labeled['label']

        imgs_weak = data_batch_unlabeled['imgs_weak']
        imgs_strong = data_batch_unlabeled['imgs_strong']

        # NOTE: ActorCutMix (replacing backgrounds)
        # Manually changed the ACM prob from 0.3 to 0.5 by Jinwoo, suggested by Yuliang
        if self.actor_cutmix and np.random.rand() < 0.5:
            human_mask = data_batch_unlabeled['human_mask']

            # NOTE: Only support num_clip = 1 case!
            human_mask = human_mask.permute(0, 4, 1, 2, 3).unsqueeze(1)
            # Select clips with valid ActorCutMix augmentation
            full_size = human_mask[0].numel()
            mask_size = human_mask.sum((1, 2, 3, 4, 5))
            invalid = torch.logical_or(mask_size == full_size, mask_size == 0)
            # If batch size is an odd number, then the central one is always invalid
            if imgs_strong.shape[0] % 2 != 0:
                invalid[imgs_strong.shape[0]//2] = True
            # For invalid clips, set the whole clip as foreground so that we won't mess up the label
            human_mask[invalid] = 1

            invalid = invalid.float().view(-1, 1, 1, 1, 1, 1)

            # When do ActorCutMix, no other aug
            fg = imgs_weak * human_mask
            bg = imgs_weak * (1-human_mask)
            bg = torch.flip(bg, dims=[0]) * (1-human_mask)
            temp = fg + bg

            imgs_strong = invalid * imgs_strong + (1-invalid) * temp

        else:
            human_mask = None

        if self.knowledge_distill:
            imagenet_prob = data_batch_unlabeled['imagenet_prob']
        else:
            imagenet_prob = None

        label_unlabeled = data_batch_unlabeled['label_unlabeled']

        losses = self(imgs, label, imgs_weak=imgs_weak, imgs_strong=imgs_strong, label_unlabeled=label_unlabeled, human_mask=human_mask, imagenet_prob=imagenet_prob)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch_labeled.values()))))

        return outputs

