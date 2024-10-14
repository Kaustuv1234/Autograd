from simplegrad.node import Node

class SGD():
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def zero_grad(self, ):
        for param in self.params:
            param.grad = 0

    def step(self, ):
        for param in self.params:
            param.value -= self.lr * param.grad