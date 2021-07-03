<center><b><font size=5>一个基于transformers的自定义命名实体识别模型示例</font></b></center>

### 一、简介

该项目是一个基于transformers的命名实体识别模型。主要目标是**提供一个代码示例，展示如果基于transformers提供的功能进行快速的模型开发。**

### 二、代码结构

+ data：包含训练数据；
+ config.py：各类参数统一定义在该文件中；
+ data_utils.py：各类数据加载的函数和全局变量定义在该文件中；
+ train_utils.py：训练过程中需要的各种辅助函数或者类定义在该文件中；
+ model.py：模型结构定义在该文件中；
+ train.py：模型训练的流程和入口在该文件中；
+ notebook.ipynb：notebook版本的代码；

### 三、运行代码

```shell
python train.py
```

### 四、本项目博客

[一个基于transformers的自定义命名实体识别模型示例](https://blog.csdn.net/bqw18744018044/article/details/118445411)



