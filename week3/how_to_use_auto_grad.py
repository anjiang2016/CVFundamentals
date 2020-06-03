#coding:utf-8
import torch


# 当x 是一个维度的时候
print("%s%s"%("-"*20,"当x是1x1的时候"))
x = torch.ones(1,1,requires_grad = True)
y = (x+2)*(x+2)*3
y.backward()
print(x.grad)


print("%s%s"%("-"*20,"当x是1x2的时候"))
x = torch.ones(1,2,requires_grad = True)
y = (x+2)*(x+2)*3
z = y.sum()
z.backward()
print(x.grad)


# 如何把autograd用到我们线性模型中
x = torch.rand(10,1)
print("%s%s"%("-"*20,"把autograd用到我们线性模型中"))
w = torch.ones(1,10,requires_grad = True)
w_init= w.data
label=20
for i in range(1,50):
    y = w.mm(x)
    loss = (y-label)*(y-label)
    loss.backward()
    #w.data = w.data - w.grad.data*0.1
    #print("loss=%s,w.requires_grad:[%s],w=[%s]"%(loss.data,w.requires_grad,w.grad))
    print("y=%s,loss=%s"%(y,loss.data))
    
    # 梯度更新写法一,use tensor.data
    #w.data = w.data - w.grad.data*0.1
    #w.grad.data = torch.zeros(1,10)
    # 梯度更新写法二,use torch.no_grad()
    with torch.no_grad():
        w -= w.grad*0.1
        w.grad.zero_()
        #pdb.set_trace()
        #w.data = w.data - w.grad.data*0.1
        #w.grad.data = torch.zeros(1,10)
    
print("*"*20)
print("model:y= wx")
print("w_init =%s"%(w_init)) 
print("y_init =%s"%(w_init.mm(x)))
print("y的目标值是%s"%(label))
print("经过梯度下降算法训练后:")
print("w=%s"%(w.data))
print("y=wx=%s"%(y.data))
