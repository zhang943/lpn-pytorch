import pytest
from copy import deepcopy

from yacs.config import CfgNode as CN

from posenet_pytorch.models import get_pose_net, PoseHighResolutionNet, PoseResNet, LightweightPoseNet
from posenet_pytorch.config import cfg, MODEL_EXTRAS


def test_get_pose_net(hrnet_config, resnet_config, lpn_config):
    hrnet = get_pose_net(hrnet_config)
    assert isinstance(hrnet, PoseHighResolutionNet)
    pose_resnet = get_pose_net(resnet_config)
    assert isinstance(pose_resnet, PoseResNet)
    lightweight_pose_net = get_pose_net(lpn_config)
    assert isinstance(lightweight_pose_net, LightweightPoseNet)


def test_get_pose_net_with_invalid_type(invalid_config):
    with pytest.raises(NameError):
        get_pose_net(invalid_config)


@pytest.fixture
def hrnet_config() -> CN:
    return _update_config_with_model_type(cfg, "pose_high_resolution_net")


@pytest.fixture
def resnet_config() -> CN:
    return _update_config_with_model_type(cfg, "pose_resnet")


@pytest.fixture
def lpn_config() -> CN:
    return _update_config_with_model_type(cfg, "lightweight_pose_net")


@pytest.fixture
def invalid_config() -> CN:
    _C = deepcopy(cfg)
    _C.defrost()
    _C.MODEL.TYPE = "invalid_type"
    _C.freeze()
    return _C


def _update_config_with_model_type(config: CN, model_type: str) -> CN:
    _C = deepcopy(config)
    _C.defrost()
    _C.MODEL.NAME = "test_" + model_type
    _C.MODEL.TYPE = model_type
    _C.MODEL.EXTRA = MODEL_EXTRAS[model_type]
    _C.freeze()
    return _C

