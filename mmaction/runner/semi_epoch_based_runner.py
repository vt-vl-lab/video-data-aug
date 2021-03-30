# Define a epoch-based runner for semi-supervised learning
import time
import warnings

import torch

import mmcv
from mmcv.runner import EpochBasedRunner, RUNNERS, get_host_info


@RUNNERS.register_module()
class SemiEpochBasedRunner(EpochBasedRunner):
    """Epoch-based Runner for semi-supervised learning.

    """

    def run_iter(self, data_batch_labeled, data_batch_unlabeled, train_mode, **kwargs):
        if train_mode:
            outputs = self.model.train_step(data_batch_labeled, data_batch_unlabeled, self.optimizer,
                                            **kwargs)
        else:
            raise NotImplementedError
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def train(self, data_loaders, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader_labeled = data_loaders[0]
        self.data_loader_unlabeled = data_loaders[1]

        # NOTE: Always use unlabeled data to count #epochs
        self._max_iters = self._max_epochs * len(self.data_loader_unlabeled)
        self.data_loader = data_loaders[1]    # Placeholder, to be compatible with mmcv

        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        iterator_labeled = iter(self.data_loader_labeled)
        for i, data_batch_unlabeled in enumerate(self.data_loader_unlabeled):
            try:
                data_batch_labeled = next(iterator_labeled)
            except:
                iterator_labeled = iter(self.data_loader_labeled)
                data_batch_labeled = next(iterator_labeled)

            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch_labeled, data_batch_unlabeled, train_mode=True)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    def val(self, data_loader, **kwargs):
        raise NotImplementedError

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for labeled
                and unlabeled data.
            workflow (list[tuple]) (deprecated): A list of (phase, epochs).
                Always [('train', 1)]
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        #assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        for _ in range(self._max_epochs):
            self.train(data_loaders, **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

