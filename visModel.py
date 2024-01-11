# -*- coding: utf-8 -*-
# @Time : 2023/12/28 21:17
# @Auth : Yuyan Yang
# @Email: youngyuyan9@163.com
# @File : visModel.py
# @Description : 可视化模型输出

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image


# 自定义评价指标
def my_acc(y_true, y_pred):
    x = 1 - (tf.sqrt(tf.square(y_pred - y_true)) / y_true)
    return tf.maximum(x, 0)


# 自定义评价指标
my_custom = {'my_acc': my_acc}

# todo
# 测试的图片路径
path = "./1.15_01.tif"
# 模型路径
my_model_path = "./results/model_1229_1049.h5"
# 找到你想要可视化的卷积层的索引
layer_names = ["conv2d"]


# 假设你已经有了一个训练好的模型
model = tf.keras.models.load_model(my_model_path, custom_objects=my_custom)
model.summary()

# 创建一个新的模型，输入与原始模型相同，输出是你想要可视化的卷积层的输出
i = 1
visualize_conv_layer_model = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[i].output)

# 使用Pillow库打开tif图片
image = Image.open(path)
# 将图片数据转换为numpy数组
image_numpy = np.array(image)
# 将numpy数组转换为TensorFlow张量
image_tensor = tf.convert_to_tensor(image_numpy)
print("原始图片大小：", image_tensor.shape)
# 缩放
x = tf.image.resize(image_tensor, [224, 224])
# 扩维，图像为3维数据，卷积层要求输入4维数据
input_image = tf.expand_dims(x, 0)

# feature_maps.shape应该是(1, height, width, num_filters)
feature_maps = visualize_conv_layer_model.predict(input_image)
# 将特征图的形状从(1, height, width, num_filters)变为(height, width, num_filters)
feature_maps = feature_maps[0]

# 获取特征图的数量
num_filters = feature_maps.shape[-1]

# 对每个特征图进行可视化
plt.figure(figsize=(10, 10))
for i in range(num_filters):
    plt.subplot(num_filters // 8 + 1, 8, i + 1)  # 创建子图
    plt.imshow(feature_maps[:, :, i], cmap='viridis')  # 显示特征图
    plt.axis('off')  # 关闭坐标轴
plt.tight_layout()  # 紧凑
plt.show()  # 显示图像
