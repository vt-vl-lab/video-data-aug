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


class SupAugBaseRecognizer(BaseRecognizer):
    """Base class for (strongly-)augmented supervised recognizers.

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
        else:
            self.actor_cutmix = False

    @abstractmethod
    def forward_train(self, imgs, labels):
        """Defines the computation performed at every call when training."""
        pass

    @abstractmethod
    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        pass

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
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
        imgs_weak = data_batch['imgs_weak']
        imgs_strong = data_batch['imgs_strong']

        # NOTE: ActorCutMix (replacing backgrounds)
        if self.actor_cutmix and np.random.rand() < 0.5:
            human_mask = data_batch['human_mask']

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
            # NOTE: In order not to modify mmcv epoch-based-runner,
            # concat human_mask with imgs_strong
            imgs_strong = torch.stack([imgs_strong, human_mask], dim=0)

        else:
            human_mask = None

        label = data_batch['label_unlabeled']
        del imgs_weak

        losses = self(imgs_strong, label)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs

