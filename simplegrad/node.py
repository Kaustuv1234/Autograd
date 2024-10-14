import math
import graphviz

class Node:

    def __init__(self, value, children=(), grad=0, op=''):
        self.value = value
        self.grad = grad
        self.children = children
        self.op = op
        self.back_pass = lambda : None

    def __repr__(self):
        return "( %.4f | %s | %.4f )" % (self.value, self.op, self.grad)

    def __add__(self, val2):
        val2 = val2 if isinstance(val2, Node) else Node(val2)
        node = Node(self.value + val2.value, (self, val2), op='+')

        def back_pass():
            for i in node.children:
                i.grad += node.grad

        node.back_pass = back_pass
        return node

    def __radd__(self, val2):
        return self + val2

    def log(self, ):
        node = Node(math.log(self.value), (self, ), op='log')

        def back_pass():
            self.grad += (1/self.value) * node.grad
        
        node.back_pass = back_pass
        return node

    def exp(self, ):
        node = Node(math.exp(self.value), (self, ), op='exp')

        def back_pass():
            self.grad += node.value * node.grad
        
        node.back_pass = back_pass
        return node

    def __mul__(self, val2):
        val2 = val2 if isinstance(val2, Node) else Node(val2)
        node = Node(self.value * val2.value, (self, val2), op='*')

        def back_pass():
            self.grad += node.grad * val2.value
            val2.grad += node.grad * self.value

        node.back_pass = back_pass
        return node

    def tanh(self):
        x = self.value
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Node(t, (self, ), op='tanh')

        def back_pass():
            self.grad += (1 - t**2) * out.grad
        out.back_pass = back_pass

        return out

    def __neg__(self):
        return self * -1

    def __pow__(self, val2):
        node = Node(self.value ** val2, (self, ), op=f'**{val2}')
        def back_pass():
            self.grad += (val2 * self.value**(val2-1)) * node.grad
        node.back_pass = back_pass
        return node

    def __truediv__(self, val2):
        val2 = val2 if isinstance(val2, Node) else Node(val2)
        node = Node(self.value / val2.value, (self, val2), op=f'÷')

        def back_pass():
            self.grad += (1/val2.value) * node.grad
            val2.grad += (-self.value/(val2.value ** 2)) * node.grad
        node.back_pass = back_pass
        return node
        # return self * val2**-1

    def __sub__(self, val2):
        return self + (-val2)

    def ReLU(self, l=0.001):
        act = lambda x : x if x >= 0 else l * x
        node = Node(act(self.value), children=(self, ), op='ReLU')

        def back_pass():
            self.grad += node.grad if node.value > 0 else (l * node.grad)

        node.back_pass = back_pass
        return node

    def sigmoid(self, ):
        act = (1 + math.exp(-self.value))**-1
        node = Node(act, children=(self, ), op='sigmoid')

        def back_pass():
            self.grad += node.grad * act * (1 - act)

        node.back_pass = back_pass
        return node

    def __rmul__(self, val2):
        return self * val2

    def backward(self):
    
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node.back_pass()


    def show_graph(self, ):
        dot = graphviz.Digraph('Network', graph_attr={'rankdir': 'LR'})
        dot.node(name=str(id(self)), label="{ %.4f | %s | %.4f }" % (self.value, self.op, self.grad), shape='record')
        visited = set()
        def create_graph(root):

            for child in root.children:
                dot.node(name=str(id(child)), label="{ %.4f | %s | %.4f }" % (child.value, child.op, child.grad), shape='record')
                dot.edge(str(id(child)), str(id(root)))
                if child in visited:
                    continue
                create_graph(child)
            visited.add(root)

        create_graph(self)
        return dot