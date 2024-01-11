# Regression-Project
A simple and general regression model training project on python.

## 中文版
程序内的路径均为【相对路径】，src_data文件夹内的数据按照组织结构放置即可。
如需更改请搜索 ./src_data, ./split_data, ./results 这几个路径并自行替换。

#-------- 当前项目文件夹组织结构 --------#
- regression
    - results - 程序运行结果
        - model.h5
        - predictions.csv
        - summary.txt
        - trainHistory.csv
        - training.png
    - split_data - 划分后的数据集
        - test_data.pkl
        - train_data.pkl
    - src_data - 原始数据
        - folder_label1 - 标签为1的图像
            - img1
            - img2
            - ...
        - folder_label2 - 标签为2的图像
        - folder_label3 - 标签为3的图像
        - ...
        - README.txt - 数据说明
    - BuildData.py - 用于创建数据集，格式为 numpy 数组
    - BuildModel.py - 用于创建未训练的模型
    - predictModel.py - 利用模型进行测试或预测
    - README.txt - 说明文档（就是你正在看的这个）
    - rowModel.h5 - 未训练的模型
    - summary.txt - 上边这个未训练模型的结构摘要
    - TrainModel.py - 用于训练模型
    - visModel.py - 用于将模型的某一层可视化输出


## English Introductions

