"""
# 数据集格式转换配置
"""
from collections import namedtuple
# 指定原始文件的XML和图片的文件夹名字
DIRECTORY_ANNOTATIONS = "Annotations/"


DIRECTORY_IMAGES = "JPEGImages/"

# 每个TFRecords文件的存储图片个数
SAMPLES_PER_FILES = 500

# VOC 2007物体类别
# VOC_LABELS = {
#     'none': (0, 'Background'),
#     'aeroplane': (1, 'Vehicle'),
#     'bicycle': (2, 'Vehicle'),
#     'bird': (3, 'Animal'),
#     'boat': (4, 'Vehicle'),
#     'bottle': (5, 'Indoor'),
#     'bus': (6, 'Vehicle'),
#     'car': (7, 'Vehicle'),
#     'cat': (8, 'Animal'),
#     'chair': (9, 'Indoor'),
#     'cow': (10, 'Animal'),
#     'diningtable': (11, 'Indoor'),
#     'dog': (12, 'Animal'),
#     'horse': (13, 'Animal'),
#     'motorbike': (14, 'Vehicle'),
#     'person': (15, 'Person'),
#     'pottedplant': (16, 'Indoor'),
#     'sheep': (17, 'Animal'),
#     'sofa': (18, 'Indoor'),
#     'train': (19, 'Vehicle'),
#     'tvmonitor': (20, 'Indoor'),
# }
# commodity_2019物体类别
VOC_LABELS = {
    'none': (0, 'Background'),
    'clothes': (1, 'clothes'),
    'pants': (2, 'pants'),
    'shoes': (3, 'shoes'),
    'watch': (4, 'watch'),
    'phone': (5, 'phone'),
    'audio': (6, 'audio'),
    'computer': (7, 'computer'),
    'books': (8, 'books'),
}

"""
# 数据集读取配置
"""
# 创建命名字典
DataSetParams = namedtuple('DataSetParameters', ['FILE_PATTERN',
                                                'NUM_CLASSES',
                                                'SPLITS_TO_SIZES',
                                                'ITEMS_TO_DESCRIPTIONS'])

# 定义commodity_2019数据属性配置
"""
FILE_PATTERN:数据集匹配字符串
NUM_CLASSES：总类别数
SPLITS_TO_SIZES：训练/测试数据集的大小
ITEMS_TO_DESCRIPTIONS：数据集读取描述信息
"""
Cm2019 = DataSetParams(
    FILE_PATTERN='commodity_2019_%s_*.tfrecord',
    NUM_CLASSES=8,
    SPLITS_TO_SIZES={
        'train': 88,
        'test': 0,
    },
    ITEMS_TO_DESCRIPTIONS={
        'image': '图片数据',
        'shape': '图篇形状',
        'object/bbox': '若干物体对象的bbox框组成的列表',
        'object/label': '若干物体的编号'
    }
)