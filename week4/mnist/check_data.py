from mnist import MNIST
import pdb
#mndata = MNIST('./dir_with_mnist_data_files')
mndata = MNIST('python-mnist/data/')
images, labels = mndata.load_training()
pdb.set_trace()
