from yacs.config import CfgNode as CN

from .lightweight_pose_net import get_pose_net as get_lightweight_pose_net
from .pose_hrnet import get_pose_net as get_pose_hrnet
from .pose_resnet import get_pose_net as get_pose_resnet

factory_methods = {
    "lightweight_pose_net": get_lightweight_pose_net,
    "pose_resnet": get_pose_resnet,
    "pose_high_resolution_net": get_pose_hrnet
}


def get_pose_net(cfg: CN, is_train: bool = False, **kwargs):
    pose_network_type = cfg.MODEL.TYPE
    if pose_network_type not in factory_methods:
        raise NameError(f"{pose_network_type} is not a valid pose network type")
    return factory_methods[pose_network_type](cfg, is_train, **kwargs)

