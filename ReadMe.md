# ReadMe

手写数字识别软件（Handwritten Digit Recognition,HDR）

## GUI

本软件GUI采用PyQt5，首先使用Qt Designer设计界面，得到HDR.ui文件

使用pyuic5命令将HDR.ui文件转换为HDR.py文件

```bash
pyuic5 -o HDR.py HDR.ui
```

## 模型

模型采用经典的LeNet网络，深度学习框架采用PyTorch，使用MNIST数据集进行训练，将训练好的模型保存下来，在软件中加载模型并对输入的图片进行预测

