#coding:utf-8
# code for week2,recognize_computer_vision.py
# houchangligong,zhaomingming,20200601,
import torch
from itertools import product
import pdb
import sys
import numpy as np

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
    # 下面添加提取图像x的特征feature的代码
    def get_shadow(x,dim):
        feature  =torch.sum(x,dim)
        feature = feature.float()
        ## 归一化
        #for i in range(0,feature.shape[0]):
        #    feature[i]=feature[i]/sum(feature)

        feature = feature.view(1,6)
        return feature
    feature  = get_shadow(x,0)
    #import pdb
    #pdb.set_trace()
    #print(feature)
    return feature
def label2ground_truth(image_label):
    gt = torch.ones(10,10)
    gt = gt*-1.0
    for label in image_label:
        gt[label,label]=float(label) 
        #gt[label,label]=float(1) 
    return gt
def model(feature,weights):
    y=-1
    # 下面添加对feature进行决策的代码，判定出feature 属于[0,1,2,3,...9]哪个类别
    #import pdb
    #pdb.set_trace()
    feature = torch.cat((feature,torch.tensor(1.0).view(1,1)),1)
    #feature2=feature.mul(feature)
    #feature3=feature2.mul(feature)
    #feature4=feature3.mul(feature)
    #pdb.set_trace()
    #y = feature.mm(weights[:,0:1])+feature2.mm(weights[:,1:2])+feature3.mm(weights[:,2:3])+feature4.mm(weights[:,3:4])
    #y = [0,-1,-1,-1]
    # y = [-1,1,-1,01]
    y = feature.mm(weights)
    return y

def train_model(image_data,image_label,weights,lr):
    loss_value_before=1000000000000000.
    loss_value=10000000000000.
    for epoch in range(0,1000):
    #epoch=0
    #while (loss_value_before-loss_value)>-1:
    
        #loss = 0 
        #for i in range(0,len(image_data)):
        loss_value_before=loss_value
        loss_value=0
        for i in range(0,10):
            #print(image_label[i])
            #y = model(get_feature(image_data[i]),weights)
            feature = get_feature(image_data[i])
            y = model(feature,weights)
            #import pdb
            #pdb.set_trace()
            gt=label2ground_truth(image_label)
            #loss = 0.5*(y-image_label[i])*(y-image_label[i])
            #loss = torch.sum((y-gt[i:i+1,:]).mul(y-gt[i:i+1,:]))
            #pdb.set_trace()
            print("%s,%s"%(y[0,i:i+1],gt[i:i+1,i:i+1]))
            loss = torch.sum((y[0,i:i+1]-gt[i:i+1,i:i+1]).mul(y[0,i:i+1]-gt[i:i+1,i:i+1]))
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
        print("epoch=%s,loss=%s/%s"%(epoch,loss_value,loss_value_before))
        #epoch+=1
        #loss_value=0
        #:loss=0
        #import pdb
        #pdb.set_trace()
    return weights
def get_result(y):
    troch.argmin(torch.from_numpy(np.numpy([torch.min((torch.abs(y-i))) for i in range(0,10)])))


if __name__=="__main__":

    weights = torch.randn(7,10,requires_grad = True)
    image_data,image_label = generate_data()
    # 打印出0的图像
    print("数字0对应的图片是:")
    print(image_data[0])
    print("-"*20)
    
    # 打印出8的图像
    print("数字8对应的图片是:")
    print(image_data[8])
    print("-"*20)
   

    lr = float(sys.argv[1])
    # 对模型进行训练：
    weights=train_model(image_data,image_label,weights,lr)


    #对每张图片进行识别
    print("对每张图片进行识别")
    for i in range(0,10):
        x=image_data[i]
        #import pdb
        #pdb.set_trace()
        #对当前图片提取特征
        feature=get_feature(x)
        # 对提取到得特征进行分类
        y = model(feature,weights)
        #pdb.set_trace()
        pred = torch.argmin(torch.from_numpy(np.array([torch.min((torch.abs(y-j))).item() for j in range(0,10)]))).item()
        #pred  = torch.argmin(torch.abs(y-1)).item()
        #打印出分类结果
        #pdb.set_trace()
        #print("图像[%s]得分类结果是:[%s],它得特征是[%s],它得二分类结果是:[%s]"%(i,torch.argmin(torch.abs(y-i)).item(),feature,y))
        print("图像[%s]得分类结果是:[%s],它得特征是[%s],它得二分类结果是:[%s]"%(i,pred,feature,y))
        

