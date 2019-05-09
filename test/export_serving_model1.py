import tensorflow as tf
from tensorflow.contrib import slim
import sys
import os
sys.path.append("../")
from nets.nets_model import ssd_vgg_300
# 加载好最新的模型图，去定义输入输出结果

data_format = "NHWC"
ckpt_filename = '../ckpt/pre_training/model.ckpt-85863'


def main(_):
    # 定义好完整的模型图，去定义输入输出结果
    # 一.
    # 1.输入：ssd模型要求的数据（不是预处理的输入）
    img_input = tf.placeholder(tf.float32, shape=(300, 300, 3))
    img_4d = image_4d = tf.expand_dims(img_input, 0)
    # 2.输出：ssd模型的输出结果
    ssd_class = ssd_vgg_300.SSDNet
    # 网络类当中参数，类别总数（商品数据集8+1）？
    ssd_params = ssd_class.default_params._replace(num_classes=9)
    # 初始化网络
    ssd_net = ssd_class(ssd_params)
    # 得出模型输出
    with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
        predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False)
    # 开启会话，加载最后保存的模型文件使得模型预测的效果达到最好
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 创建saver
        saver = tf.train.Saver()
        # 加载模型
        saver.restore(sess, ckpt_filename)
        # 二.
        # 导出模型过程
        # 路径+模型名称"./model/commodity/"
        export_path = os.path.join(
            tf.compat.as_bytes("./model/commodity/"),
            tf.compat.as_bytes(str(1))
        )
        print('正在导出模型到 to%s' % export_path)
        # 建立builder
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        # 通过该函数构建签名字典（协议）.
        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'images': tf.saved_model.utils.build_tensor_info(img_input)},
            outputs={'predict0': tf.saved_model.utils.build_tensor_info(predictions[0]),
                     'predict1': tf.saved_model.utils.build_tensor_info(predictions[1]),
                     'predict2': tf.saved_model.utils.build_tensor_info(predictions[2]),
                     'predict3': tf.saved_model.utils.build_tensor_info(predictions[3]),
                     'predict4': tf.saved_model.utils.build_tensor_info(predictions[4]),
                     'predict5': tf.saved_model.utils.build_tensor_info(predictions[5]),
                     'local0': tf.saved_model.utils.build_tensor_info(localisations[0]),
                     'local1': tf.saved_model.utils.build_tensor_info(localisations[1]),
                     'local2': tf.saved_model.utils.build_tensor_info(localisations[2]),
                     'local3': tf.saved_model.utils.build_tensor_info(localisations[3]),
                     'local4': tf.saved_model.utils.build_tensor_info(localisations[4]),
                     'local5': tf.saved_model.utils.build_tensor_info(localisations[5]),
                     },
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        #建立元图格式，写入文件
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants. SERVING],
            signature_def_map={
                'detected_model':
                    prediction_signature,
            },
            main_op=tf.tables_initializer(),
            strip_default_attrs=True)

        builder.save()
        print('完成导出!')


if __name__ == '__main__':
    tf.app.run()