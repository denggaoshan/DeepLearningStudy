import numpy as np

class LinearLayer:

    def __init__(self, input_D, output_D):
        #w，b初始值一定不能是全0，否则梯度永远是0，无法更新。
        # 参数W： (input_D X output_D)  的矩阵
        self._W = np.random.normal(0, 0.1, (input_D, output_D))
        # 参数b： (1 X output_D ) 的向量
        self._b = np.random.normal(0, 0.1, (1, output_D))
        #保存W b的梯度，W,b梯度的shape与W,b保持一致
        self._grad_W = np.zeros((input_D, output_D))
        self._grad_b = np.zeros((1, output_D))

    """
        X:  输入数据,(N X input_D）的矩阵
        返回值：    (N X output_D) 的矩阵，传往下一层
    """
    def forward(self, X):
        assert X.shape[1] == self._W.shape[0]
        return np.matmul(X, self._W) + self._b

    """
        X:    输入数据，（N X input_D）的矩阵
        grad: 反向传播过来的梯度，(N X output_D) 的向量
        返回值:梯度，(N X input_D)的矩阵，传往上一层
    """
    def backward(self, X, grad): 
        assert X.shape[0] == grad.shape[0] and X.shape[1] == self._W.shape[0] and grad.shape[1] == self._W.shape[1]
        self._grad_W = np.matmul( X.T, grad)
        self._grad_b = np.matmul(grad.T, np.ones(X.shape[0])) 
        return np.matmul(grad, self._W.T)

    """
        learn_rate: 学习率
    """   
    def update(self, learn_rate):
        self._W = self._W - self._grad_W * learn_rate
        self._b = self._b - self._grad_b * learn_rate


class Relu:
    def __init__(self):
        pass

    def forward(self, X):
        return np.where(X < 0, 0, X)

    def backward(self, X, grad):
        assert X.shape == grad.shape
        return np.where(X > 0, X, 0) * grad