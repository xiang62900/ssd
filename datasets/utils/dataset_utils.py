import tensorflow as tf


class TFRecordsReaderBase(object):
    """
    数据集基类
    """
    def __init__(self, param):
        # param给不同数据集的属性配置
        self.param = param

    def get_data(self, dataset_dir, train_or_test):
        """
        获取数据规范
        :param dataset_dir: 数据集目录
        :param train_or_test:train or test 数据文件
        :return:
        """
        return None


def int64_feature(value):
    """包裹int64型特征到Example
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """包裹浮点型特征到Example
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """包裹字节类型特征到Example
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
