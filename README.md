# CVFundamentals
  - computer vision fundamentals 
  - github:https://github.com/anjiang2016/CVFundamentals
  - week1 
  - week2
```
 CV核心基础WEEK2 ：认识计算机视觉
 Pipeline:
 1.    图像处理与计算机视觉
 2.    计算机视觉的输入与输出
 3.    如何解决计算机视觉的几个问题
 4.    计算机视觉第一步：图像描述子

 作业：
  
  编写计算机视觉的第0版程序。
  步骤
  1 生成10张图片，对应0,1,2,3,4,5,6,7,8,9.
  2 对这10张图片提取特征x。
  3 用一个判别器f(x)来决策输出结果y。
    这个判别器达到作用：
    当x是 “0”图片对应的特征时，y=f(x)=0
    当x是 “1”图片对应的特征时，y=f(x)=1
    当x是 “2”图片对应的特征时，y=f(x)=2
    当x是 “3”图片对应的特征时，y=f(x)=3
    当x是 “4”图片对应的特征时，y=f(x)=4
    当x是 “5”图片对应的特征时，y=f(x)=5
    当x是 “6”图片对应的特征时，y=f(x)=6
    当x是 “7”图片对应的特征时，y=f(x)=7
    当x是 “8”图片对应的特征时，y=f(x)=8
    当x是 “9”图片对应的特征时，y=f(x)=9
 4 参考代码:week2/recognize_computer_vision.py
```

   - week3[github:https://github.com/anjiang2016/CVFundamentals]
```
CV核心基础WEEK3 ：经典机器学习（一）
Pipeline:
1    监督学习与非监督学习
2    第一个可训练的监督学习模型：线性回归模型的3类解法
3    使用线性模型，解决字符分类问题
4    逻辑回归模型


作业：
编写计算机视觉的第1版程序：用线性回归模型，解决数字图片分类问题，
要求：用pytorch 的auto_grad功能。

步骤：
  1 生成10张图片，对应0,1,2,3,4,5,6,7,8,9.
  2 对这10张图片提取特征x。
  3 用一个线性判别器f(x)来决策输出结果y。
  4 判别器的训练要使用梯度下降法，写代码的时候要用到pytorch 的auto_grad功能。
 达到作用：
    当x是 “0”图片对应的特征时，y=f(x)=0
    ...
    当x是 “9”图片对应的特征时，y=f(x)=9
可参考代码：
  /week3/recognize_computer_vision_linear_model.py,线性模型解决图片识别问题课程代码
  /week3/how_to_use_auto_grad.py,测试pytorch auto_grad使用方法
  /week3/data_display.ipynb 数据显示
  /week3/week2作业答案课堂讲解.ipynb
  /week3/auto_grad使用时的注意事项.ipynb
  /week3/auto_grad形式的梯度下降.ipynb
  /week3/running_jupyter.pdf,jupyter运行命令
  jupyter常用效率快捷键：https://zhuanlan.zhihu.com/p/143919082
```
   - week4[github:https://github.com/anjiang2016/CVFundamentals]
```
CV核心基础WEEK4 ：经典机器学习（二）
Pipeline:
1    线性模型的局限性，以及改进方法 
2    用二分类来进行多分类：感知机
3    用逻辑回归进行多分类
4    神经网络：反向传播网络
5    【遗留】 欠拟合，过拟合与正则化
6    【遗留】 支持向量机SVM推导


作业：
已经刷爆hct66 dataset，这周就开始mnist数据集的挑战；
编写计算机视觉的的第2版程序：用3层反向传播网络来训练mnist中的100张图片。
要求：
   1 使用pytorch的auto_grad功能来编写
   2 要求随着epoch的增加，给出训练的准确度acc,和测试的准确度acc
步骤：
  1 下载并调用mnist数据集：https://github.com/anjiang2016/CVFundamentals/blob/master/week4/mnist/readme.md
  2 对mnist中的数据提取特征x。投影法可以用，但是效果不好，可以尝试用opencv 提取hog/lbp等特征
  3 用反向传播网络来决策输出结果y。
  4 反向传播网络的训练要使用梯度下降法，写代码的时候要用到pytorch 的auto_grad功能。
 期望达到精度：
    训练精度:60%
    测试精度:60%
可参考代码：
  /week4/assignment_week03_answer.py: week3作业参考答案
  /week4/recognize_computer_vision_nonlinear_model.py:用非线性模型来识别数字
  /week4/use_2class_on_multiclass.py:在htc66数据集上使用二分类进行多分类
  /week4/use_2class_on_multiclass_sigmoid.py:在htc66数据集上使用逻辑回归二分类进行多分类：w
  /week4/mnist/readme.md:mnist数据下载，读取，显示代码
  /week4/mnist_use_2class_on_multicalss.py:在mnist上使用二分类进行多分类
  /week4/mnist_bp.py:精度不好得一版bp代码
  /week4/homework_dataset: hct1000数据集【选用】
```
