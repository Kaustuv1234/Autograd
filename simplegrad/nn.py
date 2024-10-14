import random
import math
from simplegrad.node import Node

class Module():
    def __init__(self, ):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        pass


class Neuron(Module):
    def __init__(self, in_features, ):
        self.weights = [Node(random.uniform(-1.0, 1.0), op='W') for _ in range(in_features)]
        self.bias = Node(random.uniform(-1.0, 1.0), op='b')

    def forward(self, x):
        out = sum([i * j for i,j in zip(x, self.weights)], self.bias)
        # return out.ReLU(0.1)
        # return out.sigmoid()
        # return out.tanh()
        return out

    def parameters(self):
        return self.weights + [self.bias]

    def __repr__(self):
        return f"weights:{self.weights}\nbias:[{self.bias}]\n"

class Linear(Module):
    def __init__(self, in_features, out_features,):
        self.neurons = [Neuron(in_features, ) for _ in range(out_features)]

    def forward(self, x):
        out = [neur(x) for neur in self.neurons]
        return out

    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]

    def __repr__(self):
        return f"{self.neurons}"

class Sequential(Module):
    def __init__(self, *args):
        self.layers = args
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    # def parameters(self, ):
    #     return [params for layer in self.layers for params in layer.parameters()]

    def parameters(self, ):
        learn_layers = [layer for layer in self.layers if isinstance(layer, Linear)]
        return [params for layer in learn_layers for params in layer.parameters()]


class ReLU(Module):
    def __init__(self, l):
        self.l = l

    def fun(self, x):
        act = lambda x : x if x >= 0 else self.l * x
        node = Node(act(x.value), children=(x, ), op='ReLU')
        def back_pass():
            x.grad += node.grad if node.value > 0 else (self.l * node.grad)

        node.back_pass = back_pass
        return node

    def forward(self, x):
        return [self.fun(i) for i in x]


class tanh(Module):

    def fun(self, x):
        pass

    def forward(self, x):
        pass


class sigmoid(Module):

    def fun(self, x):
        pass

    def forward(self, x):
        pass

# class softmax(Module):

#     def forward(self, x):
#         denom = [math.exp]

class CrossEntropyLoss():
    def __call__(self, y, p):
        self.loss = Node(0)
        for yi, pi in zip(y, p):
            # print('yi:', yi)
            i = yi.index(1)
            self.loss += - pi[i].log()

        self.loss = self.loss * (1/len(y))
        return self.loss

    def backward(self, ):
        self.loss.backward()

class Softmax():
    def __call__(self, x):
        e = [xi.exp() for xi in x]
        denom = sum(e)
        return [ei/denom for ei in e]

class MSELoss():
    def __init__(self, ):
        pass

    def __call__(self, pred, y):
        self.loss = sum((i - j)**2 for i, j in zip(pred, y)) * (1/len(y))
        return self.loss

    def backward(self, ):
        self.loss.backward()