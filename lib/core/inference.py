# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Updated by Zhe Zhang (zhangzhe@smail.nju.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torch
import torch.nn as nn

from utils.transforms import transform_preds


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                    diff = np.array(
                        [
                            hm[py][px + 1] - hm[py][px - 1],
                            hm[py + 1][px] - hm[py - 1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals


class SoftArgmax2D(nn.Module):
    def __init__(self, height=64, width=48, beta=100):
        super(SoftArgmax2D, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.beta = beta
        # Note that meshgrid in pytorch behaves differently with numpy.
        self.WY, self.WX = torch.meshgrid(torch.arange(height, dtype=torch.float),
                                          torch.arange(width, dtype=torch.float))

    def forward(self, x):
        b, c, h, w = x.shape
        device = x.device

        probs = self.softmax(x.view(b, c, -1) * self.beta)
        probs = probs.view(b, c, h, w)

        self.WY = self.WY.to(device)
        self.WX = self.WX.to(device)

        px = torch.sum(probs * self.WX, dim=(2, 3))
        py = torch.sum(probs * self.WY, dim=(2, 3))
        preds = torch.stack((px, py), dim=-1).cpu().numpy()

        idx = np.round(preds).astype(np.int32)
        maxvals = np.zeros(shape=(b, c, 1))
        for bi in range(b):
            for ci in range(c):
                maxvals[bi, ci, 0] = x[bi, ci, idx[bi, ci, 1], idx[bi, ci, 0]]

        return preds, maxvals


def get_final_preds_using_softargmax(config, batch_heatmaps, center, scale):
    soft_argmax = SoftArgmax2D(config.MODEL.HEATMAP_SIZE[1], config.MODEL.HEATMAP_SIZE[0], beta=160)
    coords, maxvals = soft_argmax(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    batch_heatmaps = batch_heatmaps.cpu().numpy()

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                    diff = np.array(
                        [
                            hm[py][px + 1] - hm[py][px - 1],
                            hm[py + 1][px] - hm[py - 1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals
