import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from functools import partial


class MLP(nn.Module):
    def __init__(
        self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int
    ):
        super().__init__()
        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        self.layers = [
            nn.Linear(idim, odim)
            for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

    def __call__(self, x):
        for l in self.layers[:-1]:
            x = mx.tanh(l(x))
        return self.layers[-1](x)


def loss_fn(model, X, y):
    return mx.sum((model(X).squeeze() - y) ** 2)


def uncompiled_main():
    num_layers = 3
    input_dim = 3
    hidden_dim = 4
    output_dim = 1

    # Instantiate model
    model = MLP(num_layers, input_dim, hidden_dim, output_dim)

    # Dummy data
    xs = mx.array(
        [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    )
    ys = mx.array([1.0, -1.0, -1.0, 1.0])

    mx.eval(model.parameters())

    # Returns the gradient function of the loss wrt the model's trainable parameters.
    # The "from scratch" impl in main.py computes the gradient for each value, whereas this computes the gradient function itself, saving the computation for later
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.SGD(learning_rate=0.05)

    for k in range(100):
        # forward pass - computes the loss and gradients
        loss, grads = loss_and_grad_fn(model, xs, ys)

        # updates the optimizer state and the model parameters
        optimizer.update(model, grads)

        # Because MLX is compuatationally lazy, we must explicitly force updates/evals to the compute graph
        mx.eval(model.parameters(), optimizer.state)
        if k % 10 == 0:
            print(f"Epoch: {k} | Loss: {loss.item():.3f}")


def main():
    """Compiles the training loop. Roughly 30% faster training"""
    num_layers = 3
    input_dim = 3
    hidden_dim = 4
    output_dim = 1

    # Instantiate model
    model = MLP(num_layers, input_dim, hidden_dim, output_dim)

    # Dummy data
    xs = mx.array(
        [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    )
    ys = mx.array([1.0, -1.0, -1.0, 1.0])

    mx.eval(model.parameters())

    optimizer = optim.SGD(learning_rate=0.05)
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y):
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, x, y)
        optimizer.update(model, grads)
        return loss

    for k in range(100):
        loss = step(xs, ys)
        mx.eval(state)
        if k % 10 == 0:
            print(f"Epoch: {k} | Loss: {loss.item():.3f}")


def benchmark():
    import time

    start = time.perf_counter()
    uncompiled_main()
    stop = time.perf_counter()
    uncompiled_time = stop - start

    start = time.perf_counter()
    main()
    stop = time.perf_counter()
    compiled_time = stop - start

    print(f"Uncompiled time: {uncompiled_time} | Compiled time: {compiled_time}")
    print(
        f"Percent Change: {-100*((compiled_time - uncompiled_time) / uncompiled_time):.3f}%"
    )


if __name__ == "__main__":
    main()
