from .processing import ssd_vgg_preprocessing
from tensorflow.contrib import slim


preprocessing_fn_map = {
    "ssd_vgg_300": ssd_vgg_preprocessing
}
# 定义函数，预处理逻辑名，土工是否训练的处理过程


def get_preprocessing(name, is_training=True):
    """
    预处理工程获取不同的模型数据增强（预处理）方式
    :param name:模型预处理名称
    :param is_training:是否训练
    :return: 返回预处理的训练函数
    """
    if name not in preprocessing_fn_map:
        raise ValueError("选择的预处理名称 %s 不在预处理模型库当中，请提供该模型预处理代码" % name)

    # 返回一个处理的函数，后续调用这个函数
    def preprocessing_fn(image, labels, bboxes, out_shape,
                         data_format='NHWC', **kwargs):

        return preprocessing_fn_map[name].preprocess_image(image,
                                                           labels,
                                                           bboxes,
                                                           out_shape,
                                                           data_format=data_format,
                                                           is_training=is_training, **kwargs)
    return preprocessing_fn