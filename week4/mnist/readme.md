

# mnist数据集安装下载办法
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
