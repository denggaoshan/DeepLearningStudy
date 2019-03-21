import numpy as np
from Layers import LinearLayer, Relu, Sigmoid
from LossFunction import MSELoss

np.random.seed(0)

MAX_ITERATION = 10000

#用训练好的模型去预测
def predict(model, X):
    tmp = X
    for layer in model:
        tmp = layer.forward(tmp)
    return np.where(tmp > 0.5, 1, 0)

#初始化网络，总共2层，输入数据是2维，第一层3个节点，第二层1个节点作为输出层，激活函数使用Relu
linear1 = LinearLayer(2,3)
relu1 = Relu()
linear2 = LinearLayer(3,1)

#训练数据：经典的异或分类问题
train_X = np.array([[0,0],[0,1],[1,0],[1,1]])
train_y = np.array([0,1,1,0])

#开始训练网络
for i in range(MAX_ITERATION):
    #前向传播Forward，获取网络输出
    o0 = train_X
    a1 = linear1.forward(o0)
    o1 = relu1.forward(a1)
    a2 = linear2.forward(o1)
    o2 = a2

    #获得网络当前输出，计算损失loss
    result = o2.reshape(o2.shape[0])
    loss = MSELoss(train_y, result) # mean squared error loss

    #将梯度反向逐层传播，获取要更新参数的梯度
    grad = (result - train_y).reshape(result.shape[0],1)
    grad = linear2.backward(o1, grad)
    grad = relu1.backward(a1, grad)
    grad = linear1.backward(o0, grad)

    #学习率
    learn_rate = 0.01 
    #更新网络中线性层的参数
    linear1.update(learn_rate)
    linear2.update(learn_rate)

    #判断学习是否完成
    if i % 200 == 0:
        print(loss)
    if loss < 0.001:
        print("训练完成！")
        break

#将训练好的参数拼成一个网络模型
model = [linear1, relu1, linear2]

#开始预测
print("-----")
X = np.array([[0,0],[0,1],[1,0],[1,1]])
result = predict(model, X)
print("预测数据1")
print(X)
print("预测结果1")
print(result)

X = np.array([[2,2],[-1,-1],[2,-1],[-1,2]])
result = predict(model, X)
print("-----")
print("预测数据2")
print(X)
print("预测结果2")
print(result)