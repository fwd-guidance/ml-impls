import math
import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt
from graphviz import Digraph


def f(x: float | mx.array) -> float | mx.array:
    return 3 * x**2 - 4 * x + 5


def plot_fx(xs: mx.array) -> None:
    ys = f(xs)
    plt.plot(xs, ys)
    plt.show()


def fx_prime(x: float | mx.array, dx: float) -> float | mx.array:
    return (f(x + dx) - f(x)) / dx


def f_2() -> None:
    a, b, c = 2.0, -3.0, 10.0
    dx = 0.001

    d1 = a * b + c
    a = a + dx
    d2 = a * b + c

    print(f"d1 = {d1}")
    print(f"d2 = {d2}")
    print(f"slope = {(d2 - d1) / dx}")


class Value:
    data: float

    def __init__(
        self, data: float, _children: tuple = (), _op: str = "", label: str = ""
    ) -> None:
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Op only supports int/float powers"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()


def trace(root):
    """Builds  set of all nodes and edges in a graph"""
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(format="jpeg", graph_attr={"rankdir": "LR"})

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(
            name=uid,
            label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad),
            shape="record",
        )
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dot


# TODO:
def mlx_impl():
    pass
    """
    dtype = mx.float32
    x1 = mx.array([2.0], dtype=dtype)
    x2 = mx.array([0.0], dtype=dtype)
    w1 = mx.array([-3.0], dtype=dtype)
    w2 = mx.array([1.0], dtype=dtype)
    b = mx.array([6.8813735870195432], dtype=dtype)
    n = x1 * w1 + x2 * w2 + b
    o = mx.tanh(n)

    print(o.item())
    o.grad()

    print("--------")
    print("x2", x2.grad.item())
    print("w2", w2.grad.item())
    print("x1", x1.grad.item())
    print("w1", w1.grad.item())
    """


def main():
    # x = mx.arange(-5, 5, 0.25)
    # plot_fx(x)
    # a = Value(2.0, label="a")
    # b = Value(-3.0, label="b")
    # c = Value(10.0, label="c")
    # e = a * b
    # e.label = "e"
    # d = e + c
    # d.label = "d"
    # f = Value(-2.0, label="f")
    # L = d * f
    # L.label = "L"
    # inputs x1,x2
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")
    # weights w1,w2
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")
    # bias of the neuron
    b = Value(6.8813735870195432, label="b")
    # x1*w1 + x2*w2 + b
    x1w1 = x1 * w1
    x1w1.label = "x1*w1"
    x2w2 = x2 * w2
    x2w2.label = "x2*w2"
    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2.label = "x1*w1 + x2*w2"
    n = x1w1x2w2 + b
    n.label = "n"
    # o = n.tanh()
    e = (2 * n).exp()
    e.label = "e"
    o = (e - 1) / (e + 1)

    o.label = "o"
    o.backward()

    dot = draw_dot(o)
    dot.view()


if __name__ == "__main__":
    main()
