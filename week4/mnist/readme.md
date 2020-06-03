

# mnist数据集安装下载办法
```
git clone https://github.com/sorki/python-mnist
cd python-mnist
#Get MNIST data:
./bin/mnist_get_data.sh
#Get the package from PyPi:
pip install python-mnist


#调用例子
from mnist import MNIST
mndata = MNIST('./dir_with_mnist_data_files')
images, labels = mndata.load_training()



# 显示图片例子：
from mnist import MNIST
import pdb
import numpy as np
import cv2
#mndata = MNIST('./dir_with_mnist_data_files')
mndata = MNIST('python-mnist/data/')
images, labels = mndata.load_training()
#pdb.set_trace()
for img,label in zip(images,labels):
    img = np.array(img)
    img = img.reshape((28,28))
    img = img.astype('uint8')
    cv2.imshow("img",img)
    cv2.waitKey(2000)


```
