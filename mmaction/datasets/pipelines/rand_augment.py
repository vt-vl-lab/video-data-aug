# Credit: https://github.com/ildoonet/pytorch-randaugment
import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from PIL import Image

from ..registry import PIPELINES

# Set the mean pixel value as out-of-image filling value
FILL_COLOR = (124, 116, 104)


def ShearX(img, v, flip_sign, fillcolor=FILL_COLOR):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if flip_sign:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0), fillcolor=fillcolor)


def ShearY(img, v, flip_sign, fillcolor=FILL_COLOR):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if flip_sign:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0), fillcolor=fillcolor)


def TranslateX(img, v, flip_sign, fillcolor=FILL_COLOR):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if flip_sign:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0), fillcolor=fillcolor)


def TranslateXabs(img, v, flip_sign, fillcolor=FILL_COLOR):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if flip_sign:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0), fillcolor=fillcolor)


def TranslateY(img, v, flip_sign, fillcolor=FILL_COLOR):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if flip_sign:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v), fillcolor=fillcolor)


def TranslateYabs(img, v, flip_sign, fillcolor=FILL_COLOR):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if flip_sign:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v), fillcolor=fillcolor)


def Rotate(img, v, flip_sign, fillcolor=FILL_COLOR):  # [-30, 30]
    assert -30 <= v <= 30
    if flip_sign:
        v = -v
    return img.rotate(v, fillcolor=fillcolor)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def SolarizeAdd(img, addition=0, threshold=128):
    img_np = np.array(img).astype(np.int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    assert 0.05 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    assert 0.05 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.05 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.05 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v, init_loc, fillcolor=FILL_COLOR):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0, y0 = init_loc

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, fillcolor)
    return img


def Identity(img, v):
    return img


def augment_list():  # 16 oeprations and their ranges
    # https://github.com/google-research/uda/blob/master/image/randaugment/policies.py#L57
    # l = [
    #     (Identity, 0., 1.0),
    #     (ShearX, 0., 0.3),  # 0
    #     (ShearY, 0., 0.3),  # 1
    #     (TranslateX, 0., 0.33),  # 2
    #     (TranslateY, 0., 0.33),  # 3
    #     (Rotate, 0, 30),  # 4
    #     (AutoContrast, 0, 1),  # 5
    #     (Invert, 0, 1),  # 6
    #     (Equalize, 0, 1),  # 7
    #     (Solarize, 0, 110),  # 8
    #     (Posterize, 4, 8),  # 9
    #     # (Contrast, 0.1, 1.9),  # 10
    #     (Color, 0.1, 1.9),  # 11
    #     (Brightness, 0.1, 1.9),  # 12
    #     (Sharpness, 0.1, 1.9),  # 13
    #     # (Cutout, 0, 0.2),  # 14
    # ]

    # https://github.com/tensorflow/tpu/blob/8462d083dd89489a79e3200bcc8d4063bf362186/models/official/efficientnet/autoaugment.py#L505
    #l = [
    #    (AutoContrast, 0, 1),
    #    (Equalize, 0, 1),
    #    (Invert, 0, 1),
    #    (Rotate, 0, 30),
    #    (Posterize, 0, 4),
    #    (Solarize, 0, 256),
    #    (SolarizeAdd, 0, 110),
    #    (Color, 0.1, 1.9),
    #    (Contrast, 0.1, 1.9),
    #    (Brightness, 0.1, 1.9),
    #    (Sharpness, 0.1, 1.9),
    #    (ShearX, 0., 0.3),
    #    (ShearY, 0., 0.3),
    #    (CutoutAbs, 0, 40),
    #    (TranslateXabs, 0., 100),
    #    (TranslateYabs, 0., 100),
    #]

    # https://github.com/google-research/fixmatch/blob/master/libml/augment.py
    # https://arxiv.org/pdf/2001.07685.pdf - Table 12
    l = [
        (Identity, 0., 1.0),
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Rotate, 0, 30),
        (Solarize, 0, 256),
        (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95),
        (Brightness, 0.05, 0.95),
        (Sharpness, 0.05, 0.95),
        (ShearX, 0., 0.3),
        (TranslateX, 0., 0.3),
        (TranslateY, 0., 0.3),
        (Posterize, 4, 8),
        (ShearY, 0., 0.3),
        (CutoutAbs, 0, 112),
    ]

    return l


@PIPELINES.register_module()
class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30]
        self.augment_list = augment_list()

    def __call__(self, results):
        ops = random.choices(self.augment_list, k=self.n)
        # Sample once for each video clip
        flip_sign = random.random() > 0.5
        H, W, _ = results['imgs'][0].shape
        x0 = np.random.uniform(W)
        y0 = np.random.uniform(H)
        init_loc = (x0, y0)

        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            for i in range(len(results['imgs'])):
                img = Image.fromarray(results['imgs'][i])
                # NOTE: For ActorCutMix
                if 'human_mask' in results:
                    mask = Image.fromarray(results['human_mask'][i])

                if op.__name__ == 'CutoutAbs':
                    results['imgs'][i] = np.array(op(img, val, init_loc))
                    if 'human_mask' in results:
                        results['human_mask'][i] = np.array(op(mask, val, init_loc, fillcolor=0))
                elif op.__name__ in ['ShearX', 'ShearY', 'TranslateX', 'TranslateXabs', 'TranslateY', 'TranslateYabs', 'Rotate']:
                    results['imgs'][i] = np.array(op(img, val, flip_sign))
                    if 'human_mask' in results:
                        results['human_mask'][i] = np.array(op(mask, val, flip_sign, fillcolor=0))
                else:
                    results['imgs'][i] = np.array(op(img, val))

        return results
