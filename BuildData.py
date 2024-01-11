# -*- coding: utf-8 -*-
# @Time : 2023/12/27 21:58
# @Auth : Yuyan Yang
# @Email: youngyuyan9@163.com
# @File : BuildData.py
# @Description : 导入原始数据，创建、分割数据集，保存在 split_data 文件夹下

import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 源数据路径
folder_path = "./src_data(4-1)"
# 缩放后的大小
reshape_size = (224, 224)
# 训练集比例
train_ratio = 0.7

# 创建一个空列表来存储图像和标签
images = []
labels = []

# 检查文件夹是否存在
if not os.path.exists(folder_path):
    print(f"Folder {folder_path} does not exist.")

# 遍历文件夹中的所有子文件夹
for sub_folder in os.listdir(folder_path):
    sub_folder_path = os.path.join(folder_path, sub_folder)

    # 检查是否为目录
    if os.path.isdir(sub_folder_path):
        for filename in os.listdir(sub_folder_path):  # 遍历子文件夹中的所有图像文件
            if filename.endswith('.tif'):  # 修改为你的图像文件的格式
                # 读取图像文件
                img = tf.keras.preprocessing.image.load_img(
                    os.path.join(sub_folder_path, filename),
                    target_size=reshape_size)
                img = tf.keras.preprocessing.image.img_to_array(img)

                # 将子文件夹的名称转换为数值标签
                try:
                    label = float(sub_folder)  # 假设文件夹的名称可以直接转换为浮点数
                except ValueError:
                    print(f"Folder name {sub_folder} cannot be converted to a number.")
                    continue

                # 追加进数组
                images.append(img)
                labels.append(label)
        print(f"Folder {sub_folder} has been converted.")

# 将图像和标签转换为 numpy 数组
images = np.array(images)
labels = np.array(labels)

# 划分数据集为训练集和测试集集
x_train, x_test, y_train, y_test = train_test_split(images, labels, train_size=train_ratio)

# 保存为pkl文件
with open('./split_data/train_data.pkl', 'wb') as f:
    pickle.dump((x_train, y_train), f)

with open('./split_data/test_data.pkl', 'wb') as f:
    pickle.dump((x_test, y_test), f)
