import torch
import torch.nn.functional as F
from ..registry import RECOGNIZERS
# Custom imports
from .base_semi_both_aug import SemiBothAugBaseRecognizer


@RECOGNIZERS.register_module()
class SemiBothAugRecognizer3D(SemiBothAugBaseRecognizer):
    """Semi-supervised 3D recognizer model framework."""

    def forward_train(self, imgs, labels, imgs_weak, imgs_strong, labels_unlabeled, human_mask=None, imagenet_prob=None, human_mask_labeled=None):
        """Defines the computation performed at every call when training."""
        bz_labeled = imgs.shape[0]
        bz_unlabeled = imgs_weak.shape[0]
        imgs_all = torch.cat([imgs, imgs_weak, imgs_strong], dim=0)

        # TODO: If we forward imgs_weak with no_grad, and then jointly forward imgs and imgs_strong,
        # we might be able to save memory for a larger batch size? But not sure if this has
        # negative impact on batch-norm.
        imgs_all = imgs_all.reshape((-1, ) + imgs_all.shape[2:])
        x = self.extract_feat(imgs_all)
        cls_score = self.cls_head(x)

        # NOTE: pre-softmx logit
        cls_score_labeled = cls_score[:bz_labeled, :]
        cls_score_weak = cls_score[bz_labeled:bz_labeled+bz_unlabeled, :]
        cls_score_strong = cls_score[bz_labeled+bz_unlabeled:, :]

        loss = dict()

        ## Labeled data
        gt_labels = labels.squeeze()
        if self.actor_cutmix and human_mask_labeled is not None:
            temp_mask = human_mask_labeled.reshape(human_mask_labeled.shape[0], -1)
            ratio = temp_mask.sum(1) / (1. * temp_mask.shape[1])
            weight = -(ratio-1)**4+1
            weight = weight.unsqueeze(1)

            flip_gt_labels = torch.flip(gt_labels, dims=[0])
            fg_one_hot = torch.zeros((weight.shape[0], cls_score_labeled.shape[-1]), device=weight.device)
            fg_one_hot.scatter_(1, gt_labels.unsqueeze(1), 1)
            bg_one_hot = torch.zeros((weight.shape[0], cls_score_labeled.shape[-1]), device=weight.device)
            bg_one_hot.scatter_(1, flip_gt_labels.unsqueeze(1), 1)
            targets = weight * fg_one_hot + (1-weight) * bg_one_hot

            loss['loss_cls_labeled'] = -torch.mean(torch.sum(F.log_softmax(cls_score_labeled, dim=1) * targets, dim=1))

        else:
            loss_labeled = self.cls_head.loss(cls_score_labeled, gt_labels)
            loss['loss_cls_labeled'] = loss_labeled['loss_cls']
            #for k in loss_labeled:
            #    loss[k+'_labeled'] = loss_labeled[k]

        ## Unlabeled data
        with torch.no_grad():
            cls_prob_weak = F.softmax(cls_score_weak, dim=-1)

        # TODO: Control this threshold with cfg
        if self.framework == 'fixmatch':
            thres = 0.95
        else:    # UDA
            thres = 0.8
        select = (torch.max(cls_prob_weak, 1).values >= thres).nonzero(as_tuple=False).squeeze(1)
        num_select = select.shape[0]
        if num_select > 0:
            if self.framework == 'fixmatch':
                all_pseudo_labels = cls_score_weak.argmax(1).detach()
                pseudo_labels = torch.index_select(all_pseudo_labels.view(-1, 1), dim=0, index=select).squeeze(1)
            else:    # UDA
                # UDA uses soft (sharpened) pseudo labels
                all_pseudo_labels = F.softmax(cls_score_weak / 0.4, dim=1)
                pseudo_labels = torch.index_select(all_pseudo_labels, dim=0, index=select)
            cls_score_unlabeled = torch.index_select(cls_score_strong, dim=0, index=select)

            # NOTE: When we have ActorCutMix, we should do label smoothing
            # We use the human tube ratio (wrt to the whole clip) to determine the weight
            if self.actor_cutmix and human_mask is not None:
                temp_mask = human_mask.reshape(human_mask.shape[0], -1)
                temp_mask = torch.index_select(temp_mask, dim=0, index=select)
                ratio = temp_mask.sum(1) / (1. * temp_mask.shape[1])
                weight = -(ratio-1)**4+1
                weight = weight.unsqueeze(1)

                flip_pseudo_labels = torch.flip(all_pseudo_labels, dims=[0])
                if self.framework == 'fixmatch':
                    flip_pseudo_labels = torch.index_select(flip_pseudo_labels.view(-1, 1), dim=0, index=select)
                    fg_one_hot = torch.zeros((weight.shape[0], cls_score_weak.shape[-1]), device=weight.device)
                    fg_one_hot.scatter_(1, pseudo_labels.unsqueeze(1), 1)
                    bg_one_hot = torch.zeros((weight.shape[0], cls_score_weak.shape[-1]), device=weight.device)
                    bg_one_hot.scatter_(1, flip_pseudo_labels, 1)
                    targets = weight * fg_one_hot + (1-weight) * bg_one_hot
                else:    # UDA
                    flip_pseudo_labels = torch.index_select(flip_pseudo_labels, dim=0, index=select)
                    targets = weight * pseudo_labels + (1-weight) * flip_pseudo_labels

                loss_unlabeled = dict()
                loss_unlabeled['loss_cls'] = -torch.mean(torch.sum(F.log_softmax(cls_score_unlabeled, dim=1) * targets, dim=1))
            else:
                if self.framework == 'fixmatch':
                    loss_unlabeled = self.cls_head.loss(cls_score_unlabeled, pseudo_labels)
                else:    # UDA
                    loss_unlabeled = dict()
                    loss_unlabeled['loss_cls'] = -torch.mean(torch.sum(F.log_softmax(cls_score_unlabeled, dim=1) * pseudo_labels, dim=1))

            # NOTE: When do loss reduce_mean, we should always divide by the batch size
            # instead of number of confident samples. So we compensate here.
            if self.framework == 'fixmatch':
                loss['loss_cls_unlabeled'] = loss_unlabeled['loss_cls'] * num_select / bz_unlabeled
            else:
                loss['loss_cls_unlabeled'] = loss_unlabeled['loss_cls']

            ## Get statistics for strong/weak pair for sanity check
            #labels_unlabeled = torch.index_select(labels_unlabeled, dim=0, index=select).squeeze(1)
            #cls_score_weak = torch.index_select(cls_score_weak, dim=0, index=select)
            #cls_score_strong = torch.index_select(cls_score_strong, dim=0, index=select)
            #with torch.no_grad():
            #    loss_weak = self.cls_head.loss(cls_score_weak, labels_unlabeled)
            #    loss_strong = self.cls_head.loss(cls_score_strong, labels_unlabeled)
            #loss['top1_acc_weak'] = loss_weak['top1_acc']
            #loss['top1_acc_strong'] = loss_strong['top1_acc']
            #loss['top5_acc_weak'] = loss_weak['top5_acc']
            #loss['top5_acc_strong'] = loss_strong['top5_acc']

        else:
            loss['loss_cls_unlabeled'] = torch.zeros(1).to(cls_prob_weak.device)
            #loss['top1_acc_weak'] = torch.zeros(1).to(cls_prob_weak.device)
            #loss['top1_acc_strong'] = torch.zeros(1).to(cls_prob_weak.device)
            #loss['top5_acc_weak'] = torch.zeros(1).to(cls_prob_weak.device)
            #loss['top5_acc_strong'] = torch.zeros(1).to(cls_prob_weak.device)

        with torch.no_grad():
            loss['num_select'] = (torch.ones(1)*num_select).to(cls_score_weak.device)

        # NOTE: Knowledge distillation from ImageNet pre-trained model
        # Probably need to multiply a weighting factor
        if self.knowledge_distill:
            x_weak = x[bz_labeled:bz_labeled+bz_unlabeled, :]
            distill_score_weak = self.distill_head(x_weak)
            loss['loss_distill'] = -torch.mean(torch.sum(F.log_softmax(distill_score_weak, dim=1) * imagenet_prob, dim=1))

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
        raise NotImplementedError
