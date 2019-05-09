from datasets.dataset_init import commodity_2019
from datasets.dataset_config import Cm2019
# 逻辑
# 1.数据集名称
# 2.指定训练集还是测试集
# 3.数据集目录的指定.

# 定义dataset种类的字典:名称：class()
datasets_map = {
    'commodity_2019': commodity_2019.CommodityTFRecords
}


def get_datasets(dataset_name, train_or_test, dataset_dir):
    """
    获取不同数据集数据
    :param name: 数据集名字
    :param train_or_test: 指定训练集还是测试集
    :param dataset_dir:数据集目录的指定.
    :return:Dataset 数据规范
    """

    if dataset_name not in datasets_map:
        raise ValueError("输入的数据集名称%s 不存在" % dataset_name)
    # print()
    # print(train_or_test)
    return datasets_map[dataset_name](Cm2019).get_data(train_or_test, dataset_dir)