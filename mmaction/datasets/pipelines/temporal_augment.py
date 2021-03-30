import numpy as np

from ..registry import PIPELINES


@PIPELINES.register_module()
class TemporalHalf(object):
    """Drop half of the frames in a clip, and fill with the other half.

    Required keys are "imgs", modified keys are "imgs".

    Args:
        prob (float): Probability of applying this operation.
    """

    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, results):
        """Perform the TemporalHalf operation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if results['num_clips'] > 1:
            raise NotImplementedError

        if np.random.rand() > self.prob:
            return results

        l = results['clip_len']
        if np.random.rand() < 0.5:
            results['imgs'][0:l//2] = results['imgs'][l//2:l]
            if 'human_mask' in results:
                results['human_mask'][0:l//2] = results['human_mask'][l//2:l]
        else:
            results['imgs'][l//2:l] = results['imgs'][0:l//2]
            if 'human_mask' in results:
                results['human_mask'][l//2:l] = results['human_mask'][0:l//2]
        return results


@PIPELINES.register_module()
class TemporalReverse(object):
    """Reverse the clip in the temporal-axis.

    Required keys are "imgs", modified keys are "imgs".

    Args:
        prob (float): Probability of applying this operation.
    """

    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, results):
        """Perform the TemporalHalf operation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if results['num_clips'] > 1:
            raise NotImplementedError

        if np.random.rand() > self.prob:
            return results

        results['imgs'] = results['imgs'][::-1]
        if 'human_mask' in results:
            results['human_mask'] = results['human_mask'][::-1]

        return results


@PIPELINES.register_module()
class TemporalCutOut(object):
    """Randomly drop a frame and fill with its previou frame.

    Required keys are "imgs", modified keys are "imgs".

    Args:
        prob (float): Probability of applying this operation (per-frame).
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, results):
        """Perform the TemporalCutOut operation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if results['num_clips'] > 1:
            raise NotImplementedError

        l = results['clip_len']
        drops = np.random.rand(l) < self.prob
        # Assume we always fill with the previous frame
        drops[0] = False
        aug_idx = list(range(l))
        for i, drop in enumerate(drops):
            if drop:
                aug_idx[i] = aug_idx[i-1]

        out = [results['imgs'][i] for i in aug_idx]
        results['imgs'] = out
        if 'human_mask' in results:
            out = [results['human_mask'][i] for i in aug_idx]
            results['human_mask'] = out

        return results


@PIPELINES.register_module()
class TemporalAugment(object):
    """Randomly apply temporal augmentations.

    Required keys are "imgs", modified keys are "imgs".

    Args:
        prob (float): Probability of applying this operation.
    """

    def __init__(self, prob=0.25):
        self.prob = prob
        self.temporal_half = TemporalHalf(prob=1.0)
        self.temporal_reverse = TemporalReverse(prob=1.0)
        self.temporal_cutout = TemporalCutOut(prob=0.5)

    def __call__(self, results):
        """Perform the TemporalCutOut operation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if results['num_clips'] > 1:
            raise NotImplementedError

        rnd = np.random.rand()
        if rnd < 0.25:
            return results
        elif rnd < 0.5:
            return self.temporal_half(results)
        elif rnd < 0.75:
            return self.temporal_reverse(results)
        else:
            return self.temporal_cutout(results)

