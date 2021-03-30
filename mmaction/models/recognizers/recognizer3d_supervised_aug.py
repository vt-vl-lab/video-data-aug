import torch
import torch.nn.functional as F

from ..registry import RECOGNIZERS
# Custom imports
from .base_supervised_aug import SupAugBaseRecognizer


@RECOGNIZERS.register_module()
class SupAugRecognizer3D(SupAugBaseRecognizer):
    """3D recognizer model framework."""

    def forward_train(self, imgs, labels):
        """Defines the computation performed at every call when training."""
        # NOTE: Get human_mask
        if len(imgs.shape) == 7:
            human_mask = imgs[1]
            imgs = imgs[0]
        else:
            human_mask = None

        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        x = self.extract_feat(imgs)
        cls_score = self.cls_head(x)
        gt_labels = labels.squeeze()
        loss = dict()
        if self.actor_cutmix and human_mask is not None:
            temp_mask = human_mask.reshape(human_mask.shape[0], -1)
            ratio = temp_mask.sum(1) / (1. * temp_mask.shape[1])
            weight = -(ratio-1)**4+1
            weight = weight.unsqueeze(1)

            flip_gt_labels = torch.flip(gt_labels, dims=[0])
            fg_one_hot = torch.zeros((weight.shape[0], cls_score.shape[-1]), device=weight.device)
            fg_one_hot.scatter_(1, gt_labels.unsqueeze(1), 1)
            bg_one_hot = torch.zeros((weight.shape[0], cls_score.shape[-1]), device=weight.device)
            bg_one_hot.scatter_(1, flip_gt_labels.unsqueeze(1), 1)
            targets = weight * fg_one_hot + (1-weight) * bg_one_hot

            loss['loss_cls'] = -torch.mean(torch.sum(F.log_softmax(cls_score, dim=1) * targets, dim=1))

        else:
            loss_labeled = self.cls_head.loss(cls_score, gt_labels)
            loss['loss_cls'] = loss_labeled['loss_cls']

        return loss

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        x = self.extract_feat(imgs)
        cls_score = self.cls_head(x)
        cls_score = self.average_clip(cls_score)

        return cls_score.cpu().numpy()

    def forward_dummy(self, imgs):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        x = self.extract_feat(imgs)
        outs = (self.cls_head(x), )
        return outs
