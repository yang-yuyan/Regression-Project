# -*- coding: utf-8 -*-
# @Time : 2023/12/28 20:41
# @Auth : Yuyan Yang
# @Email: youngyuyan9@163.com
# @File : predictModel.py
# @Description : 测试模型

import csv
import pickle
import datetime
import numpy as np
import tensorflow as tf


# 自定义评价指标
def my_acc(y_true, y_pred):
    x = 1 - (tf.sqrt(tf.square(y_pred - y_true)) / y_true)
    return tf.maximum(x, 0)


# 自定义评价指标
def r2(y_true, y_pred):
    y_true_mean = tf.math.reduce_mean(y_true)
    corr_numerator = tf.math.reduce_sum(tf.math.square(y_true - y_pred))
    corr_denominator = tf.math.reduce_sum(tf.math.square(y_true - y_true_mean))
    correlation_coefficient = 1 - (corr_numerator / corr_denominator)

    return correlation_coefficient


# 自定义评价指标
# custom_name = 'my_acc'
# my_custom = {custom_name: my_acc}
custom_name = 'correlation_coefficient'
my_custom = {custom_name: r2}


# 测试集路径
test_path = './split_data/train_data.pkl'
# 批次大小
batch_size = 8
# todo 模型路径
my_model_path = './results/model_1229_1049.h5'

# 测试集
with open(test_path, 'rb') as f:
    x_test, y_test = pickle.load(f)
    # 制作成 tf_dataset 格式并设置 batch_size
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

# 导入模型
my_model = tf.keras.models.load_model(my_model_path, custom_objects=my_custom)

# 预测
predictions = my_model.predict(test_dataset).flatten()
print(predictions)
# 获取一个批次的大小
batch_size = len(next(iter(test_dataset))[1])
# 时间戳 月日_时分
nowTime = datetime.datetime.now().strftime('%m%d_%H%M')
# 保存预测结果为csv文件
with open(f"./results/predictions_{nowTime}.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    # 写入表头
    writer.writerow(['True', 'Prediction'])
    # 写入数据和预测结果
    for (_, labels), prediction in zip(test_dataset, np.array_split(predictions, len(predictions) // batch_size)):
        writer.writerows(zip(labels.numpy(), prediction))
