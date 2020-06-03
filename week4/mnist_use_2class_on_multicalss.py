#coding:utf-8
# code for week2,recognize_computer_vision.py
# houchangligong,zhaomingming,20200602,
import torch
from itertools import product
import pdb
import sys
from mnist import MNIST
import cv2
import numpy as np
#mndata = MNIST('python-mnist/data/')
#images, labels = mndata.load_training()
def generate_data():
    # 本函数生成0-9，10个数字的图片矩阵
    image_data=[]
    num_0 = torch.tensor(
    [[0,0,1,1,0,0],
    [0,1,0,0,1,0],
    [0,1,0,0,1,0],
    [0,1,0,0,1,0],
    [0,0,1,1,0,0],
    [0,0,0,0,0,0]])
    image_data.append(num_0)
    num_1 = torch.tensor(
    [[0,0,0,1,0,0],
    [0,0,1,1,0,0],
    [0,0,0,1,0,0],
    [0,0,0,1,0,0],
    [0,0,1,1,1,0],
    [0,0,0,0,0,0]])
    image_data.append(num_1)
    num_2 = torch.tensor(
    [[0,0,1,1,0,0],
    [0,1,0,0,1,0],
    [0,0,0,1,0,0],
    [0,0,1,0,0,0],
    [0,1,1,1,1,0],
    [0,0,0,0,0,0]])
    image_data.append(num_2)
    num_3 = torch.tensor(
    [[0,0,1,1,0,0],
    [0,0,0,0,1,0],
    [0,0,1,1,0,0],
    [0,0,0,0,1,0],
    [0,0,1,1,0,0],
    [0,0,0,0,0,0]])
    image_data.append(num_3)
    num_4 = torch.tensor(
    [
    [0,0,0,0,1,0],
    [0,0,0,1,1,0],
    [0,0,1,0,1,0],
    [0,1,1,1,1,1],
    [0,0,0,0,1,0],
    [0,0,0,0,0,0]])
    image_data.append(num_4)
    num_5 = torch.tensor(
    [
    [0,1,1,1,0,0],
    [0,1,0,0,0,0],
    [0,1,1,1,0,0],
    [0,0,0,0,1,0],
    [0,1,1,1,0,0],
    [0,0,0,0,0,0]])
    image_data.append(num_5)
    num_6 = torch.tensor(
    [[0,0,1,1,0,0],
    [0,1,0,0,0,0],
    [0,1,1,1,0,0],
    [0,1,0,0,1,0],
    [0,0,1,1,0,0],
    [0,0,0,0,0,0]])
    image_data.append(num_6)
    num_7 = torch.tensor(
    [
    [0,1,1,1,1,0],
    [0,0,0,0,1,0],
    [0,0,0,1,0,0],
    [0,0,0,1,0,0],
    [0,0,0,1,0,0],
    [0,0,0,0,0,0]])
    image_data.append(num_7)
    num_8 = torch.tensor(
    [[0,0,1,1,0,0],
    [0,1,0,0,1,0],
    [0,0,1,1,0,0],
    [0,1,0,0,1,0],
    [0,0,1,1,0,0],
    [0,0,0,0,0,0]])
    image_data.append(num_8)
    num_9 = torch.tensor(
    [[0,0,1,1,1,0],
    [0,1,0,0,1,0],
    [0,1,1,1,1,0],
    [0,0,0,0,1,0],
    [0,0,0,0,1,0],
    [0,0,0,0,0,0]])
    image_data.append(num_9)
    image_label=[0,1,2,3,4,5,6,7,8,9]
    return image_data,image_label
    
def get_feature(x):

    feature=[0,0,0,0]
    xa = np.array(x)
    xt = torch.from_numpy(xa.reshape(28,28))
    # 下面添加提取图像x的特征feature的代码
    def get_shadow(x,dim):
        feature  =torch.sum(x,dim)
        feature = feature.float()
        ## 归一化
        for i in range(0,feature.shape[0]):
            feature[i]=feature[i]/sum(feature)

        feature = feature.view(1,28)
        return feature
    #pdb.set_trace()
    feature  = get_shadow(xt,0)
    #import pdb
    #pdb.set_trace()
    #print(feature)
    return feature
def label2ground_truth(image_label):
    gt = torch.ones(10,10)
    gt = gt*-1.0
    #for label in image_label:
    for i in range(0,10):
        gt[i,i]=float(image_label[i]) 
    return gt
def model(feature,weights):
    y=-1
    # 下面添加对feature进行决策的代码，判定出feature 属于[0,1,2,3,...9]哪个类别
    #import pdb
    #pdb.set_trace()
    feature = torch.cat((feature,torch.tensor(1.0).view(1,1)),1)
    feature2=feature.mul(feature)
    #feature3=feature2.mul(feature)
    #feature4=feature3.mul(feature)
    #pdb.set_trace()
    #y = feature.mm(weights[:,0:1])+feature2.mm(weights[:,1:2])+feature3.mm(weights[:,2:3])+feature4.mm(weights[:,3:4])
    y = feature.mm(weights)
    #y = 1.0/(1.0+torch.exp(-1.*h))
    return y
def one_hot(gt):
    gt_vector = torch.ones(1,10)
    gt_vector *= -1.0*0.1
    gt_vector[0,gt] = 1.0*0.9
    return gt_vector
def get_acc(image_data,image_label,weights,start_i,end_i):

    correct=0
    for i in range(start_i,end_i):
             #print(image_label[i])
             #y = model(get_feature(image_data[i]),weights)
             feature = get_feature(image_data[i])
             y = model(feature,weights)
             #pdb.set_trace()
             gt = image_label[i]
             #pred=torch.argmin(torch.abs(y-gt)).item()
             pred = torch.argmin(torch.from_numpy(np.array([torch.min((torch.abs(y-j))).item() for j in range(0,10)]))).item()
             #pred = torch.argmin((torch.abs(y-1))).item()
             #print("图像[%s]得分类结果是:[%s]"%(gt,pred))
             if gt==pred:
                 correct+=1
             
    #print("acc=%s"%(float(correct/20.0)))
    return  float(correct/float(end_i-start_i))
def train_model(image_data,image_label,weights,lr):
    loss_value_before=1000000000000000.
    loss_value=10000000000000.
    for epoch in range(0,3000):
    #epoch=0
    #while (loss_value_before-loss_value)>-1:
    
        #loss = 0 
        #for i in range(0,len(image_data)):
        loss_value_before=loss_value
        loss_value=0
        for i in range(0,80):
            #print(image_label[i])
            #y = model(get_feature(image_data[i]),weights)
            feature = get_feature(image_data[i])
            y = model(feature,weights)
            #import pdb
            #pdb.set_trace()
            #gt=label2ground_truth(image_label)
            #loss = 0.5*(y-image_label[i])*(y-image_label[i])
            #loss = torch.sum((y-gt[i:i+1,:]).mul(y-gt[i:i+1,:]))
            #pdb.set_trace()
            gt = image_label[i]
            # 只关心一个值
            loss = torch.sum((y[0,gt:gt+1]-gt).mul(y[0,gt:gt+1]-gt))
            #gt_vector = one_hot(gt)
            #pdb.set_trace()
            # 关心所有值
            #loss = torch.sum((y-gt_vector).mul(y-gt_vector))+0.0*torch.sum(weights.mul(weights))
            # 用log的方式
            #pdb.set_trace()
            #loss = -torch.log(y[0,gt])-torch.sum(torch.log(1.0-y[0,0:gt]))-torch.sum(torch.log(1-y[0,gt:-1]))
            # 优化loss，正样本接近1，负样本远离1
            #loss1 = (y-1.0).mul(y-1.0)
            #loss = loss1[0,gt]+torch.sum(1.0/(loss1[0,0:gt]))+torch.sum(1.0/(loss1[0,gt:-1]))
            #print("%s,%s"%(y[0,gt:gt+1],gt))
            #pdb.set_trace()
            #print("%s,%s"%(torch.argmin(torch.abs(y-1)).item(),gt))
            #loss.data.add_(loss.data) 
            loss_value += loss.data.item()
            #print("loss=%s"%(loss))
            #weights =
            # 更新公式
            # w  = w - (y-y1)*x*lr
            #feature=feature.view(6)
            #lr=-lr
            #weights[0,0] = weights[0,0]+ (y.item()-image_label[i])*feature[0]*lr
            #weights[1,0] = weights[1,0]+ (y.item()-image_label[i])*feature[1]*lr
            #weights[2,0] = weights[2,0]+ (y.item()-image_label[i])*feature[2]*lr
            #weights[3,0] = weights[3,0]+ (y.item()-image_label[i])*feature[3]*lr
            #weights[4,0] = weights[4,0]+ (y.item()-image_label[i])*feature[4]*lr
            #weights[5,0] = weights[5,0]+ (y.item()-image_label[i])*feature[5]*lr
            #weights[6,0] = weights[6,0]+ (y.item()-image_label[i])*lr
            loss.backward()
            weights.data.sub_(weights.grad.data*lr)
            weights.grad.data.zero_()
            #loss.data=
        #import pdb
        #print("epoch=%s,loss=%s/%s,weights=%s"%(epoch,loss_value,loss_value_before,(weights[:,0:2]).view(14)))
        train_acc = get_acc(image_data,image_label,weights,0,80)
        test_acc = get_acc(image_data,image_label,weights,80,100)
        print("epoch=%s,loss=%s/%s,train/test_acc=%s/%s,"%(epoch,loss_value,loss_value_before,train_acc,test_acc))
        #epoch+=1
        #loss_value=0
        #:loss=0
        #import pdb
        #pdb.set_trace()
    return weights

if __name__=="__main__":

    weights = torch.randn(29,10,requires_grad = True)
    # hct66 dataset , 10 samples
    image_data,image_label = generate_data()
    # minst 2828 dataset 60000 samples
    mndata = MNIST('./mnist/python-mnist/data/')
    image_data_all, image_label_all = mndata.load_training()
    #import pdb
    #pdb.set_trace()
    image_data=image_data_all[0:100]
    image_label=image_label_all[0:100]
    ''' 
    pdb.set_trace()
    
    # 打印第1张图像
    print("数字%s对应的图片是:"%(image_label[0]))
    #print(image_data[3])
    #cv2.imshow("name",img)
    cv2.imshow(str(image_label[0]),np.array(image_data[0]).reshape((28,28)).astype('uint8'))
    # wait 2000 ms
    cv2.waitKey(20000)
    print("-"*20)
    
    # 打印出第2张图像
    print("数字%s对应的图片是:"%(image_label[1]))
    cv2.imshow(str(image_label[1]),np.array(image_data[1]).reshape((28,28)).astype('uint8'))
    cv2.waitKey(20000)
    print("-"*20)
    '''

    lr = float(sys.argv[1])
    # 对模型进行训练：
    weights=train_model(image_data,image_label,weights,lr)

    


    #测试：
    correct=0
    for i in range(0,10):
             #print(image_label[i])
             #y = model(get_feature(image_data[i]),weights)
             feature = get_feature(image_data[i])
             y = model(feature,weights)
             #pdb.set_trace()
             gt = image_label[i]
             #pred=torch.argmin(torch.abs(y-gt)).item()
             pred = torch.argmin(torch.from_numpy(np.array([torch.min((torch.abs(y-j))).item() for j in range(0,10)]))).item()
             #pred = torch.argmin(torch.abs(y-1)).item()
             print("图像[%s]得分类结果是:[%s]"%(gt,pred))
             if gt==pred:
                 correct+=1
             
    print("acc=%s"%(float(correct/10.0)))
