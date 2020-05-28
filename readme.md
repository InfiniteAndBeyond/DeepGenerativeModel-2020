# 深度生成模型2020
## 简介
根据[FoldingNet](https://arxiv.org/abs/1712.07262v2)得到的一些启发，做的一个实验。将2d平面换成了3d球面  
同时做了一些将2d图片映射到3d球面的实验。

## 环境
* Linux
* python>=3.6
* pytorch >=1.2,<1.5
* visdom: 用于显示实验结果
* [kaolin](https://github.com/NVIDIAGameWorks/kaolin): 用于计算chamfer distance

## 代码说明
* reference：存放参考代码
  * dataset_shapenet_completion.py：参考代码，用于数据集读取
  
* model：存放模型
  * Folding.py: 修改的folding net模型，以及2d图片映射到3d球面等物体的小实验
  * common.py：folding net中使用到的层
* train_foldingNet.py：训练代码
* utliz.py：用于各类显示

## 数据集
ShapeNet Completion，下载链接:[Google Drive](https://drive.google.com/open?id=1M_lJN14Ac1RtPtEQxNlCV9e8pom3U6Pa)

## 查重说明
* reference文件夹为参考代码，很多复制的部分
* 查重仅需要查model中的两个py文件(Folding.py, common.py)，和根目录下的两个py文件(train_foldingNet.py, utliz.py)，共4个文件
  



