# coding:utf-8
# code for week3

import torch
from torch.autograd import Variable as V

def generate_data():
    # 本函数生成0-9，10个数字的图片矩阵
    image_data = []
    num_0 = torch.tensor(
        [[0, 0, 1, 1, 0, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0]])
    image_data.append(num_0)
    num_1 = torch.tensor(
        [[0, 0, 0, 1, 0, 0],
         [0, 0, 1, 1, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0]])
    image_data.append(num_1)
    num_2 = torch.tensor(
        [[0, 0, 1, 1, 0, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0, 0],
         [0, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0]])
    image_data.append(num_2)
    num_3 = torch.tensor(
        [[0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0]])
    image_data.append(num_3)
    num_4 = torch.tensor(
        [
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0]])
    image_data.append(num_4)
    num_5 = torch.tensor(
        [
            [0, 1, 1, 1, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0]])
    image_data.append(num_5)
    num_6 = torch.tensor(
        [[0, 0, 1, 1, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 1, 1, 1, 0, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0]])
    image_data.append(num_6)
    num_7 = torch.tensor(
        [
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0]])
    image_data.append(num_7)
    num_8 = torch.tensor(
        [[0, 0, 1, 1, 0, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 0, 1, 1, 0, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0]])
    image_data.append(num_8)
    num_9 = torch.tensor(
        [[0, 0, 1, 1, 1, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 0, 1, 1, 1, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0]])
    image_data.append(num_9)
    image_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    return image_data, image_label

def get_feature(x, dim):
    # 下面添加提取图像x的特征feature的代码
    Heigh = x.shape[0]
    feature = torch.sum(x, dim)
    feature = feature.float()
    feat_dim = feature.shape[0]
    #  归一化
    for i in range(0, feat_dim):
        feature[i] = feature[i] / sum(feature)
    feature = feature.view(1, Heigh)
    return feature

def train_model(weights, learning_rate, iters, num_data, image_data, image_label):

    for epoch in range(iters):
        loss = 0
        for i in range(0, num_data):
            feature = get_feature(image_data[i], 1)
            y_pred = linear_model(feature, weights)
            loss += 0.5 * (y_pred - image_label[i])**2
        # 自动计算梯度
        loss.backward()
        # 更新参数
        weights.data.sub_(learning_rate * weights.grad.data)
        # 梯度清零，不清零梯度会累加
        weights.grad.data.zero_()
        print('each epoch loss is {}'.format(loss.item()))
    return weights

def linear_model(feature, weights):
    y = -1
    feature = torch.cat((feature, torch.tensor(1.0).view(1,1)), 1)
    y = feature.mm(weights)
    return y


if __name__ == "__main__":

    image_data, image_label = generate_data()
    num_sample = len(image_data)
    num_feat = 6
    # 初始化随机参数
    weights = torch.rand(num_feat + 1, 1, requires_grad=True)
    learning_rate = 0.05
    iters = 5000
    num_data = 6
    new_weights = train_model(weights, learning_rate, iters, num_data, image_data, image_label)

    print("对每张图片进行识别")
    for i in range(0, num_sample):
        x = image_data[i]
        # 对当前图片提取特征
        dim = 0
        feature = get_feature(x, dim)
        # 对提取到得特征进行分类
        y = linear_model(feature, weights)
        # 打印出分类结果
        print("图像[%s]得分类结果是:[%s]" % (i, y))

