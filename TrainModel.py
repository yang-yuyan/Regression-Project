# -*- coding: utf-8 -*-
# @Time : 2023/12/28 14:36
# @Auth : Yuyan Yang
# @Email: youngyuyan9@163.com
# @File : TrainModel.py
# @Description : 训练模型，结果保存在同级目录下的 results 文件夹下

import csv
import sys
import pickle
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


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
custom_name = 'r2'
my_custom = {custom_name: r2}


# 绘制训练过程
def show_loss(history):
    # 从 history 中提取模型训练集和验证集准确率信息和误差信息
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # 画图
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('mean_squared_error')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')

    # 时间戳 月日_时分
    nowTime = datetime.datetime.now().strftime('%m%d_%H%M')
    # 保存图片
    plt.savefig(f"./results/training_{nowTime}.png", dpi=100)


# 训练集路径
train_path = './split_data/train_data.pkl'
# 批次大小
batch_size = 8

# todo 模型路径
my_model_path = 'rowModel.h5'
# 交叉验证折数
n_splits = 3
# 训练轮次
epoch = 80

# 导入数据集
with open(train_path, 'rb') as f:
    x_train, y_train = pickle.load(f)
# 创建 KFold 对象
kf = KFold(n_splits)
# 初始化一些参数
acc_list = []
loss_list = []
best_loss = float('inf')
best_model = None
best_history = None

# 时间戳 月日_时分
nowTime = datetime.datetime.now().strftime('%m%d_%H%M')

# 每循环一折
for train_index, val_index in kf.split(x_train):
    # 获取训练集和验证集
    x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    # 制作成 tf_dataset 格式并设置 batch_size
    train_fold = tf.data.Dataset.from_tensor_slices((x_train_fold, y_train_fold)).batch(batch_size)
    val_fold = tf.data.Dataset.from_tensor_slices((x_val_fold, y_val_fold)).batch(batch_size)

    # 导入模型
    my_model = tf.keras.models.load_model(my_model_path, custom_objects=my_custom)
    # 训练并计时
    begin_time = datetime.datetime.now()
    my_history = my_model.fit(train_fold, validation_data=val_fold, epochs=epoch)
    end_time = datetime.datetime.now()
    print(f"训练时间：{end_time - begin_time}s")  # 该循环程序运行时间： 1.4201874732

    # 输出这个模型的 验证集 自定义评价指标
    acc = my_history.history[f'val_{custom_name}'][-1]
    acc_list.append(acc)
    # 输出这个模型的 验证集 损失
    val_loss = my_history.history['val_loss'][-1]
    loss_list.append(val_loss)

    # 最佳模型判断：如果这个模型的损失比之前的模型低，就保存这个模型和它的损失
    if val_loss < best_loss:
        best_loss = val_loss
        best_model = my_model
        best_history = my_history
print(f"验证集损失列表：\n {loss_list}")
print(f"验证集自定义评价指标列表：\n {acc_list}")

# 保存并绘制最佳模型损失曲线
show_loss(best_history)

# 保存最佳训练历史
with open(f"./results/trainHistory_{nowTime}.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    for key, value in best_history.history.items():
        writer.writerow([key] + value)

# 保存最佳模型
best_model.save(f"./results/model_{nowTime}.h5")

# 保存模型摘要
sys.stdout = open(f"./results/summary_{nowTime}.txt", 'w')
best_model.summary()
sys.stdout.close()
sys.stdout = sys.__stdout__
