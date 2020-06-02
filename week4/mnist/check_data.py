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


