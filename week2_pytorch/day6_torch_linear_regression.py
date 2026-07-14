import torch


def create_dataset(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    创建模拟线性回归数据。

    真实规律：
        y = 3x + 2 + noise
    """
    x = torch.linspace(-2, 2, 200, device=device).reshape(-1, 1)

    noise = torch.randn(200, 1, device=device) * 0.3
    y = 3 * x + 2 + noise

    return x, y


def train_linear_regression(
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 1000,
    learning_rate: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    使用 autograd 训练线性回归模型。
    """
    device = x.device

    # 模型参数，需要计算梯度
    weight = torch.randn(
        1,
        1,
        device=device,
        requires_grad=True,
    )

    bias = torch.zeros(
        1,
        device=device,
        requires_grad=True,
    )

    for epoch in range(1, epochs + 1):
        # 1. 前向传播
        prediction = x @ weight + bias

        # 2. 计算均方误差
        loss = ((prediction - y) ** 2).mean()

        # 3. 反向传播，计算梯度
        loss.backward()

        # 4. 根据梯度更新参数
        with torch.no_grad():
            weight -= learning_rate * weight.grad
            bias -= learning_rate * bias.grad

        # 5. 清空上一轮梯度
        weight.grad.zero_()
        bias.grad.zero_()

        if epoch == 1 or epoch % 100 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"Loss: {loss.item():.6f} | "
                f"Weight: {weight.item():.4f} | "
                f"Bias: {bias.item():.4f}"
            )

    return weight, bias


def predict(
    x: float,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> float:
    """
    使用训练完成的参数进行预测。
    """
    input_tensor = torch.tensor(
        [[x]],
        dtype=torch.float32,
        device=weight.device,
    )

    with torch.no_grad():
        prediction = input_tensor @ weight + bias

    return prediction.item()


def main() -> None:
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"使用设备: {device}")

    x, y = create_dataset(device)

    print("训练数据 shape:")
    print("x:", x.shape)
    print("y:", y.shape)

    weight, bias = train_linear_regression(
        x=x,
        y=y,
        epochs=1000,
        learning_rate=0.05,
    )

    print("\n训练完成")
    print(f"学习到的 weight: {weight.item():.4f}")
    print(f"学习到的 bias: {bias.item():.4f}")
    print("真实 weight: 3.0000")
    print("真实 bias: 2.0000")

    test_x = 4.0
    prediction = predict(test_x, weight, bias)

    print(f"\n输入 x = {test_x}")
    print(f"模型预测 y = {prediction:.4f}")
    print(f"理论结果 y = {3 * test_x + 2:.4f}")


if __name__ == "__main__":
    main()