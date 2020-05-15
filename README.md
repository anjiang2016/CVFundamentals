# CVFundamentals
computer vision fundamentals 
```
week1
week2
week3
```
章节	周数	题目	关键字	详情	课堂可用的案例	作业	项目
基础篇 I 计算机视觉基础	1	课程导论及初阶计算机视觉 I (Low-Level CV)	基础图像处理	"
图像读写 图像基本属性:图像大小、维度、数据类型、通道、颜色空间等 图像 ROI图像γ值转换图像相似性变换 图像仿射变换
图像投影变换。   图像一阶、二阶导数
图像卷积
图像导数、卷积的关系与应用:边缘检测、锐化、模糊 。角点检测"		作业 实现数据增广、自己实现给图片加隐形水印	
	2	认识计算机视觉 	计算机视觉的中阶与高阶。	2.1) 特征点与特征描述子:SIFT 2.2) SIFT 程序演示  2.3）  HoG特征，原理及应用  2.4 RANSAC算法介绍与参数推导 ）2.5) 传统 CV 流程  		  作业 ：RANSAC实现仿射变换的单应矩阵计算 	项目1:   图像拼接
 基础篇 II 经典机器学习 	3	经典机器学习 I:线性回归与逻辑回归 		"3.1 机器学习简介 
3.1.1) 监督学习 
3.1.2) 非监督学习 
3.2 线性回归 
3.2.1) 形式与定义3.2.2) 损失函数3.2.3) 梯度下降与推导
3.2.4) Normal Equation 3.2.5) 线性回归的程序演示 
3.3 逻辑回归 
3.3.1) 形式与定义 3.3.2) Sigmoid 函数 3.3.3) 梯度下降与推导 3.3.4) 简单的分类问题 "		"作业 ：动手实现一个线性回归模型。
 "	
	4	经典机器学习 II:神经网络、反向传播算法以及正则化 		"神经网络(Neural Networks) 
4.1.1) 神经元及神经网络实例 4.1.2) 神经网络形式化表述 4.1.3) 神经网络的程序演示 
反向传播算法(Backpropagation) 
4.2.1) 反向传播算法实例 4.2.2) 反向传播算法形式化表述 4.2.3) 反向传播算法的问题 
正则化(Regularization) 
4.3.1) 正则化目的及原理 4.3.2) 正则化推导及实现 4.3.3) L1 与 L2 正则化对比 
"		"作业 ：动手实现一个反向传播
"	
	5	经典机器学习 III:其他机器学习工具及总结 		"5.1 支持向量机(SVM) 
5.1.1) 基础 SVM 推导 
5.1.2) SVM 与 Soft Margin 5.1.3) SVM 与 Kernel 
5.2非监督学习(Unsupervised Learning) 
5.2.1) K-Means 算法5.2.2) K-Means 算法问题及 K-Means++算法 5.2.3) K-Means 算法与 kNN 算法5.2.4) K-Means 算法程序展示 
概念与总结 
5.3.1) Bias 与 Variance5.3.2) Overfit 与 Underfit 概念、成因与解决 5.3.3) 梯度消失与爆炸5.3.4) 训练、验证与测试集 
决策树(Decision Tree) 
5.4.1) ID3算法:熵(Entropy)、条件熵(ConditionalEntropy)与增益(Info-Gain) 5.4.2) C4.5 算法:增益比(Gain Ratio)5.4.3) CART 算法:基尼系数(Gini Index) "		"作业 
5.5.1) Coding:实现 K-Means++算法"	项目 2：  人脸关键点检测 
	6	CNN 综述 I:层 	卷积神经网络是什么，为什么要这样，具体内部结构的种类，作用	"CNN 综述 
6.1.1) CNN 与传统 CV 关系 6.1.2) CNN 总体流程 
CNN 基础层 
6.2.1)  卷积层: 卷积层参数、实现、转置卷积、空洞卷积、反向传播、C3D 等 
6.2.2)  ReLU 层:ReLU, PReLU, Leaky ReLU 
6.2.3)  池化(Pooling)层:Average Pooling & Max Pooling 
6.2.4)  全连接 (FC/Inner Product) 层 
CNN 功能性层 
6.3.1)  Batch Normalization:Batch Normalization 推导、理解、含义与训练 
6.3.2)  Dropout 层:Dropout 层目的、推导与实现 "		"作业 
6.4.1) 源码阅读与程序学习:PyTorch 基础层搭建
"	
	7	CNN 综述 III:网络架构	CNN网络架构的历史发展和经典架构	"网络架构历史发展 
7.1.1)  网络架构历史发展 LeNet-5 AlexNet ZFNet 
7.1.2)  经典网络架构范式 
经典网络架构 
7.2.1) VGG: 感受野(Receptive Field)及计算机7.2.2) Inception Net: 1x1 Conv, Module 以及 Ensemble 效果 7.2.3) ResNet: Bottleneck 以及 Shortcut7.2.4) DenseNet: 
轻型网络结构 
7.3.1)  轻型网络发展历史及思路总结 
7.3.2)  SqueezeNet:Fire Module 
7.3.3)  MobileNet-V1:Depthwise Convolution MobileNet-V2:Inverted Residuals & Linear Bottlenecks 
7.3.4)  ShuffleNet-V1:Group Conv ShuffleNet-V2:4 Guidelines 及计算证明 
7.4 其他网络结构 
7.4.1) 其他经典网络结构 
7.5 FLOPs 
7.5.1)  FLOPs 的概念 
7.5.2)  FLOPs 计算与实例 "		作业：pytorch 网络搭建	
	8	CNN 综述 III:实现细节 	卷积神经网络的参数初始化和梯度下降算法进行优化的方式	"网络参数初始化策略 
8.1.1) Gaussian / Xavier / Maiming 含义与推导 
图像预处理 
8.2.1) 传统图像预处理8.2.2) 主成分分析(PCA)推导 8.2.3) CNN 图像预处理 
参数优化方式 
8.3.1)  基于动量的方式的含义与推导 SGD SGD + Momentum Nesterov 
8.3.2)  自适应方式的含义与推导: Adagrad RMSProp Adam 
8.3.3)  后 Adam 方法 AdaMax & Nadam 
8.3.4)  参数优化总结 
评价方式 
7.4.1) 召回率/准确率/精确度 7.4.2) AP/ROC 
学习策略 
7.5.1) 学习率的变化 "		9.3.1)  动物多分类实战 （作业）	
	9	cuda编程		"9.1, GPU Schema  9.2, CUDA 的 Thread，Block 与 Grids     9.3, Memory 层次结构 ,· 9.4,  CUDA 变量，从 1 block and 1 thread 到 N blocks and N threads
9. 5, 使用 CUDA 实现 Matrix Multiplication"			"项目3 基于深度学习垃圾图片智能检测分类项目 
"
应用篇 卷积神经网络各类应用 	10	 CNN 的分类问题	分类问题	"分类问题大纲 
10.1.1) 二分类问题10.1.2) 多分类问题 (Multi-Class Classification):Softmax 定义与求导 9.1.3) 多标签分类 (Multi-Label Classification)10.1.4) 多任务分类 (Multi-Task Classification) 
分类问题的实际问题 
10.2.1) 多分类/标签问题:动物分类项目讲解 10.2.2) 不均衡数据问题:数据、loss、算法层面讲解 10.2.3) 细粒度分类问题:特征、注意力机制等讲解 
"			
	11	 CNN 的检测问题 I:Two-Stage 检测算法 	两阶段目标检测算法	"10.1 RCNN 算法 
10.1.1) RCNN 算法的实现细节 
10.1.2) NMS 系列的发展 
10.2 Fast RCNN 
10.2.1) Fast RCNN 算法的实现细节 
10.2.2) ROI Pooling 系列的发展 
10.3 Faster RCNN 
10.3.1) Faster RCNN 算法的实现细节 10.3.2) RPN 网络10.3.3) Anchor "			
	12	 CNN 的检测问题 II:One-Stage 检测算法 	一阶段目标检测算法	"11.3 Yolo V3 
11.3.1)  FPN 网络 
11.3.2)  Yolo V3 算法的实现细节 11.3.3）先验框与统计框的设计 11.3.4）对预测边框的约束策略 11.3.5）Passthrough 11.3.6）绘制 YOLO v3 的计算流 11.3.7） COCO 数据集与数据处理 11.3.8）解析 GT 生成策略

11.4.1) Focal Loss 的原理与应用 
11.5 其他算法 
11.5.1) SSD 系列11.5.2) 当前无 Anchor 趋势 "		作业:探索 YOLO v3 与 FCN 在输出结 构上的融合思路	
	13	图像分割 	分割问题	"12.1 图像分割 
12.1.1) 像素分类思想 、反卷积与升采样、跳级结构、FCN12.1.2) UNet/ENet 12.1.3) Mask RCNN 12.1.4) 图像分割的发展 
"			"项目4:街景中的车牌识别项目 
"
	14	目标跟踪		14.1）为什么会有跟踪任务  14.2）跟踪的方法（TLD / KCF）14.3）卡尔曼滤波 的原理与代码			
