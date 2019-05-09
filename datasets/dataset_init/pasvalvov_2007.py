import tensorflow as tf
from tensorflow.contrib import slim
import os


def get_dataset(dataset_dir):
    """
    获取pascalvoc2007数据集
    :param dataset_dir: 数据集目录
    :return: Dataset
    """
    # 1.构造第一个参数：数据目录+文件名
    file_pattern = os.path.join(dataset_dir, "VOC_2007_test_*.tfrecord")
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
                 num_samples=4952,
                 items_to_descriptions={
                            'image': 'A color image of varying height and width.',
                            'shape': 'Shape of the image',
                            'object/bbox': 'A list of bounding boxes, one per each object.',
                            'object/label': 'A list of labels, one per each object.'
                                },  # 数据的描述说明，字典
                 num_classes=20)
