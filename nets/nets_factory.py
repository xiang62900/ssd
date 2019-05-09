from nets.nets_model import ssd_vgg_300
# from object_detection.models.research.object_detection.models import ssd_mobilenet_v2_fpn_feature_extractor
from tensorflow.contrib import slim


networks_obj = {
                'ssd_vgg_300': ssd_vgg_300.SSDNet,
                # 'ssd_mobilenet_v2':ssd_mobilenet_v2_fpn_feature_extractor.SSDMobileNetV2FpnFeatureExtractor,
                }


def get_network(netword_name):
    """
    获取模型网络实例
    :param netword_name:网络模型的名称
    :return:
    """
    return networks_obj[netword_name]