import pytest
from typing import Type, List, Tuple
import random

from yacs.config import CfgNode as CN
import numpy as np
import torch
import torch.nn as nn

from posenet_pytorch.models import LightweightPoseNet
from posenet_pytorch.models.lightweight_modules import LW_Bottleneck
from posenet_pytorch.utils.inference import get_final_preds
from posenet_pytorch.config.default import _C as DEFAULT_CONFIG
from posenet_pytorch.config.models import MODEL_EXTRAS


class TestLightweightPoseNet:
    def test_init(self, block: Type[LW_Bottleneck], layers: List[int], config: CN):
        model = LightweightPoseNet(block, layers, config)
        assert isinstance(model, LightweightPoseNet)
        assert isinstance(model, nn.Module)

    def test_forward(
            self,
            model: LightweightPoseNet,
            config: CN,
            patches: torch.Tensor,
            batch_size: int,
            centers: List[Tuple[float, float]],
            scales: List[Tuple[float, float]]
        ):
        model.init_weights()
        model_output = model(patches).clone().cpu().detach().numpy()
        final_preds, _ = get_final_preds(config, model_output, centers, scales)
        assert final_preds.shape == (batch_size, config.MODEL.NUM_JOINTS, 2)


@pytest.fixture
def config() -> CN:
    _C = DEFAULT_CONFIG
    _C.MODEL.EXTRA = MODEL_EXTRAS['lightweight_pose_net']
    return _C


@pytest.fixture
def block() -> Type[LW_Bottleneck]:
    return LW_Bottleneck


@pytest.fixture
def layers() -> List[int]:
    return [3, 4, 6, 3]


@pytest.fixture
def model(block, layers, config) -> LightweightPoseNet:
    return LightweightPoseNet(block, layers, config)


@pytest.fixture
def batch_size() -> int:
    return random.randint(16, 32)


@pytest.fixture
def patches(batch_size, config) -> torch.Tensor:
    patches_shape = (batch_size, 3, *config.MODEL.IMAGE_SIZE)
    return torch.randint(0, 256, patches_shape).type(torch.float)


@pytest.fixture
def centers(batch_size) -> List[Tuple[float]]:
    image_size = (640, 360)
    centers_ = np.random.random((batch_size, 2))
    for dim in range(2):
        centers_[:, dim] = centers_[:, dim] * image_size[dim]
    return centers_


@pytest.fixture
def scales(batch_size) -> List[Tuple[float]]:
    return np.random.random((batch_size, 2)) * 2
