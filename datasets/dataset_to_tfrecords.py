import os
import tensorflow as tf
import xml.etree.ElementTree as et

from .dataset_config import DIRECTORY_ANNOTATIONS, DIRECTORY_IMAGES, SAMPLES_PER_FILES, VOC_LABELS
from .utils.dataset_utils import int64_feature, float_feature, bytes_feature


def _get_out_filename(outpudir, name, fdx):
    """
    获取输出的文件名
    :param outpudir: 输出文件路径
    :param name: 数据集名字voc_2007_test or voc_2007_train
    :param fix:第几个批次0~
    :return:
    """
    return "%s/%s_%03d.tfrecord" % (outpudir, name, fdx)


def _process_image(dataset_dir, img_name):
    """
    处理一张图片及xml的逻辑
    :param dataset_dir:数据集目录
    :param img_name:该图片名字编号
    :return:
        """
    # 1.处理图片
    # 构造图片的文件名字
    filename = dataset_dir + DIRECTORY_IMAGES + img_name+".jpg"
    # 读取图片
    img_data = tf.gfile.FastGFile(filename, 'rb').read()
    # 2.处理xml
    filename_xml = dataset_dir + DIRECTORY_ANNOTATIONS + img_name + ".xml"
    # 需要使用ET工具读取,将文件内容变成树装结构
    tree = et.parse(filename_xml)
    root = tree.getroot()
    # 获取size信息
    size = root.find('size')
    # 把三个宽、高、通道数存在一个shape里
    shape = [int(size.find('height').text),
            int(size.find('width').text),
            int(size.find('depth').text)]
    # 获取object信息
    # 每个object都包含name，truncated，difficult，bndbox[xmin，ymin，xmax，ymax]
    objects = root.findall('object')
    labels = []
    labels_text = []
    difficults = []
    truncateds = []
    bboxes = []
    for obj in objects:
        # 解析每一个obj
        # 取出目标label，具体的物体类别
        # 动物：猫、狗等等  数字与之一一对应
        label = obj.find('name').text
        labels_text.append(label.encode('utf-8'))

        # 取出与之对应的物体大类别，存入编号
        labels.append(int(VOC_LABELS[label][0]))
        # truncated
        if obj.find('truncated'):
            truncateds.append(int(obj.find('truncated').text))
        else:
            truncateds.append(0)

        # 取出diffcult
        if obj.find('difficult'):
            difficults.append(int(obj.find('difficult').text))
        else:
            difficults.append(0)
        # 取出每个对象的四个坐标值
        bbox = obj.find('bndbox')
        # xmin,ymin,xmax,ymax都要进行除以原图片的长宽
        bboxes.append((float(bbox.find('ymin').text) / shape[0],
                       float(bbox.find('xmin').text) / shape[1],
                       float(bbox.find('ymax').text) / shape[0],
                       float(bbox.find('xmax').text) / shape[1]))
    return img_data, shape, labels, labels_text, difficults, truncateds, bboxes


def _convert_to_example(img_data, shape, labels, labels_text, difficults, truncateds, bboxes):
    """"
    一张图片数据封装，使用example 封装成protobufer格式
    :param image_data: 图片内容
    :param shape: 图片形状
    :param bboxes: 每一个目标的四个位置值
    :param difficult: 默认0
    :param truncated: 默认0
    :param labels: 目标类别代号
    :return:
    """
    # 为了转换需求对bboxes进行格式调整，从一个obj的四个属性列表，变成四个位置单独的列表
    # [[12,23,34,45], [56,23,76,9]] --->ymin [12, 56], xmin [23, 23], ymax [34, 76], xmax [45, 9]
    ymin = []
    xmin = []
    ymax = []
    xmax = []
    for box in bboxes:
        ymin.append(box[0])
        xmin.append(box[1])
        ymax.append(box[2])
        xmax.append(box[3])
    # 将所有信息封装成example
    # 每一个存入的属性都要指定类型
    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(shape[0]),
        'image/width': int64_feature(shape[1]),
        'image/channels': int64_feature(shape[2]),
        'image/shape': int64_feature(shape),
        'image/object/bbox/xmin': float_feature(xmin),
        'image/object/bbox/xmax': float_feature(xmax),
        'image/object/bbox/ymin': float_feature(ymin),
        'image/object/bbox/ymax': float_feature(ymax),
        'image/object/bbox/label': int64_feature(labels),
        'image/object/bbox/label_text': bytes_feature(labels_text),
        'image/object/bbox/difficult': int64_feature(difficults),
        'image/object/bbox/truncated': int64_feature(truncateds),
        'image/format': bytes_feature(image_format),
        'image/encoded': bytes_feature(img_data)}))

    return example

def _add_to_tfrecord(dataset_dir, img_name, tfrecord_writer):
    """
    # 1、读取图片内容以及图片相对应的XML文件
    # 2、读取的内容封装成example, 写入指定tfrecord文件
    :param dataset_dir: 数据集目录
    :param img_name: 该图片名字
    :param tfrecord_writer: 写入文件实例
    :return:
    """
    # 1.读取每张图片的内容，以及每张图片对应的xml内容
    img_data, shape, labels, labels_text, difficults, truncateds, bboxes = _process_image(dataset_dir, img_name)
    # print(img_data)
    # 2.将每张图片封装成example
    example = _convert_to_example(img_data, shape, labels, labels_text, difficults, truncateds, bboxes)
    # 3.tfrecord_writer写入example的序列列化结果
    tfrecord_writer.write(example.SerializeToString())
    return


def run(dataset_dir, outputdir, name='data'):
    """
    运行转换代码逻辑，存入多个tfrecord文件，每个文件固定N个样本
    :param dataset_dir:数据集目录
    :param outputdir:TFRecotds存储目录
    :param name:数据集名字，指定名字以及train or test
    :return:
    """
    # 1.判断目录是否存在，不存在创建目录
    if tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)
    # 2.去读某个文件夹下的所有文件名字列表
    path = os.path.join(dataset_dir, DIRECTORY_ANNOTATIONS)
    # 读取所有文件时候会打乱顺序,要排序
    filenames = sorted(os.listdir(path))
    # 3.循环这个名字列表
    # 每个500个图片已经XML信息就存储到一个tfrecord文件当中
    # 4.每500章图片及xml信息就存储到一个TFRecotd文件中
    i = 0
    fdx = 0
    while i < len(filenames):
        # 1.创建一个TFRecord文件，有第几个文件的序号
        tf_file_name = _get_out_filename(outputdir, name, fdx)
        # 2.循环标记每隔500图片内容，存储一次
        # 新建一个TFRecord文件存储器
        with tf.python_io.TFRecordWriter(tf_file_name) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j<SAMPLES_PER_FILES:
                print("转换图片进度%d/%d" % (i + 1, len(filenames)))
                # 取出图片的名字及xml的名字
                # single_filename *.xml
                single_filename = filenames[i]
                img_name = single_filename[: -4]
                # 读取图片内容以及xml文件内容，存入文件
                # 默认每次构造一个图片文件存储指定文件
                # 每个图片构造Iexample存储到tf_filename当中
                _add_to_tfrecord(dataset_dir, img_name, tfrecord_writer)
                i += 1
                j += 1
            # 存储文件的喜欢也要进行增加
            fdx += 1
    print("完成数据集 %s 所有的样本处理" % name)
    return