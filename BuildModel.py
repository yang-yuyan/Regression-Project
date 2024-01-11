# -*- coding: utf-8 -*-
# @Time : 2023/12/28 15:53
# @Auth : Yuyan Yang
# @Email: youngyuyan9@163.com
# @File : BuildModel.py
# @Description : 构建模型，保存未训练的模型及结构

import sys
import tensorflow as tf
from tensorflow.keras import models, layers


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


# vgg16
def vgg16():
    model = models.Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(224, 224, 3)),
        # block1
        layers.Conv2D(16, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'),
        layers.Conv2D(16, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), padding='same'),

        # block2
        layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'),
        layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), padding='same'),

        # block3
        layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'),
        layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'),
        layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), padding='same'),

        # block4
        layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'),
        layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'),
        layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), padding='same'),

        # full-connect
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        # layers.Dropout(0.5),
        layers.Dense(8, activation='linear'),
        # layers.Dropout(0.5),
        layers.Dense(1, activation='linear')  # regression
    ])
    return model


# 全连接
def fc_nn():
    model = models.Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(224, 224, 3)),
        layers.Conv2D(5, (7, 7), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'),
        # layers.BatchNormalization(),
        layers.Conv2D(15, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'),
        layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
        layers.Conv2D(15, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'),
        layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
        layers.Conv2D(15, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'),
        layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
        layers.Conv2D(15, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'),
        layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),  # full-connect
        # layers.Dropout(0.5),
        layers.Dense(2, activation='relu'),  # full-connect
        # layers.Dropout(0.5),
        layers.Dense(1, activation='linear')  # regression
    ])
    return model


if __name__ == '__main__':
    # 加载模型
    my_model = fc_nn()

    # 编译
    my_model.compile(optimizer='adam', loss='mse', metrics=[r2])  # 自定义评价指标

    # 保存模型
    my_model.save("rowModel.h5")
    # 保存模型摘要
    sys.stdout = open('summary.txt', 'w')
    my_model.summary()
    sys.stdout.close()
    sys.stdout = sys.__stdout__
