import tensorflow as tf
from tensorflow.contrib import slim
import os
from datasets.utils.dataset_utils import TFRecordsReaderBase


class CommodityTFRecords(TFRecordsReaderBase):
    """
    商品数据集读取类
    """
    def __init__(self, param):
        self.param = param

    def get_data(self, train_or_test, dataset_dir):
        """
        获取pascalvoc2007数据集
        :param dataset_dir: 数据集目录
        :param train_or_test:train or test 数据文件
        :return: Dataset
        """
        # 异常抛出
        # print(train_or_test)
        if train_or_test not in ['train', 'test']:
            raise ValueError("训练/测试数据集的名字%s指定错误" % train_or_test)
        # 判断数据集目录
        if not tf.gfile.Exists(dataset_dir):
            raise ValueError("数据集的目录错误")

        # 1.构造第一个参数：数据目录+文件名
        file_pattern = os.path.join(dataset_dir, self.param.FILE_PATTERN % train_or_test)
        # 2.准备第二个参数
        reader = tf.TFRecordReader
        # 3.准备第三个参数：decoder
        # 1、反序列化成数据原来的格式
        keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/height': tf.FixedLenFeature([1], tf.int64),
            'image/width': tf.FixedLenFeature([1], tf.int64),
            'image/channels': tf.FixedLenFeature([1], tf.int64),
            'image/shape': tf.FixedLenFeature([3], tf.int64),
            'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
            'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
            'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64),
        }
        # 2、反序列化成高级的格式
        items_to_handlers = {
            'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
            'shape': slim.tfexample_decoder.Tensor('image/shape'),
            'object/bbox': slim.tfexample_decoder.BoundingBox(
                ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
            'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
            'object/difficult': slim.tfexample_decoder.Tensor('image/object/bbox/difficult'),
            'object/truncated': slim.tfexample_decoder.Tensor('image/object/bbox/truncated'),
        }

        # 构造decoder
        decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features=keys_to_features,
                                                          items_to_handlers=items_to_handlers)
        return slim.dataset.Dataset(
                     data_sources=file_pattern,
                     reader=reader,
                     decoder=decoder,
                     num_samples=self.param.SPLITS_TO_SIZES[train_or_test],
                     items_to_descriptions=self.param.ITEMS_TO_DESCRIPTIONS,  # 数据的描述说明，字典
                     num_classes=self.param.NUM_CLASSES)
