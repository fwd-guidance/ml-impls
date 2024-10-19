from nn import MLP
import matplotlib.pyplot as plt
from graphviz import Digraph


def f(x: float) -> float:
    """Default mathematical function"""
    return 3 * x**2 - 4 * x + 5


def fx_prime(x: float, dx: float) -> float:
    """'Theoretical' definition of the derivative"""
    return (f(x + dx) - f(x)) / dx


def f_2() -> None:
    """'Empirical' definition of the derivative"""
    a, b, c = 2.0, -3.0, 10.0
    dx = 0.001

    d1 = a * b + c
    a = a + dx
    d2 = a * b + c

    print(f"d1 = {d1}")
    print(f"d2 = {d2}")
    print(f"slope = {(d2 - d1) / dx}")


def plot_fx(xs: float) -> None:
    ys = f(xs)
    plt.plot(xs, ys)
    plt.show()


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
    """Saves .jpeg diagram of the computational graph"""
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


def test_mlp():
    x = [2.0, 3.0, -1.0]
    n = MLP(3, [4, 4, 1])
    print(n(x))
    # mlp_dot = draw_dot(n(x))
    # mlp_dot.view()


def main():
    n = MLP(3, [4, 4, 1])
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys: list[float] = [1.0, -1.0, -1.0, 1.0]

    for k in range(100):
        # forward pass
        ypred = [n(x) for x in xs]
        loss = sum(
            (y_out - y_ground_truth) ** 2 for y_ground_truth, y_out in zip(ys, ypred)
        )

        # backward pass
        for p in n.parameters():
            p.grad = 0.0
        loss.backward()

        # update
        for p in n.parameters():
            p.data += -0.05 * p.grad

        print(k, loss.data)


if __name__ == "__main__":
    main()
