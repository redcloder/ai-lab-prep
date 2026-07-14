import torch


def basic_derivative() -> None:
    print("=== 示例 1：基本求导 ===")

    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 2

    y.backward()

    print("x:", x)
    print("y:", y)
    print("x.grad:", x.grad)


def chain_rule() -> None:
    print("\n=== 示例 2：链式法则 ===")

    x = torch.tensor(2.0, requires_grad=True)

    a = x * 3
    b = a + 1
    y = b ** 2

    y.backward()

    print("a:", a)
    print("b:", b)
    print("y:", y)
    print("x.grad:", x.grad)


def loss_gradient() -> None:
    print("\n=== 示例 3：loss 对参数求梯度 ===")

    weight = torch.tensor(2.0, requires_grad=True)
    bias = torch.tensor(1.0, requires_grad=True)

    x = torch.tensor(3.0)
    target = torch.tensor(10.0)

    prediction = weight * x + bias
    loss = (prediction - target) ** 2

    loss.backward()

    print("prediction:", prediction)
    print("loss:", loss)
    print("weight.grad:", weight.grad)
    print("bias.grad:", bias.grad)


def gradient_accumulation() -> None:
    print("\n=== 示例 4：梯度累积 ===")

    x = torch.tensor(2.0, requires_grad=True)

    y1 = x ** 2
    y1.backward()

    print("第一次 backward:", x.grad)

    x.grad.zero_()

    y2 = x ** 2
    y2.backward()

    print("清空后再次 backward:", x.grad)


def vector_gradient() -> None:
    print("\n=== 示例 5：向量与标量 loss ===")

    x = torch.tensor(
        [1.0, 2.0, 3.0],
        requires_grad=True,
    )

    y = x ** 2
    loss = y.mean()

    loss.backward()

    print("y:", y)
    print("loss:", loss)
    print("x.grad:", x.grad)


def main() -> None:
    basic_derivative()
    chain_rule()
    loss_gradient()
    gradient_accumulation()
    vector_gradient()


if __name__ == "__main__":
    main()