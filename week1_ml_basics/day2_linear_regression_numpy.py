import numpy as np
import matplotlib.pyplot as plt


def main():
    # 1. 造一批简单数据
    # 真实规律：y = 3x + 2 + 噪声
    np.random.seed(42)

    X = np.linspace(0, 10, 100)
    noise = np.random.randn(100) * 2
    y = 3 * X + 2 + noise

    # 2. 初始化参数
    w = 0.0
    b = 0.0

    # 3. 设置训练参数
    learning_rate = 0.003
    epochs = 4000
    n = len(X)

    losses = []

    # 4. 开始训练
    for epoch in range(epochs):
        # 预测
        y_pred = w * X + b

        # 计算 MSE loss
        loss = np.mean((y_pred - y) ** 2)
        losses.append(loss)

        #打印信息
        if epoch < 10 or (epoch<2000 and epoch % 100==0) or epoch%1000==0 :
            print(f"before update | epoch: {epoch}, loss: {loss:.4f}, w: {w:.4f}, b: {b:.4f}")

        # 计算梯度
        dw = (2 / n) * np.sum((y_pred - y) * X)
        db = (2 / n) * np.sum(y_pred - y)

        # 梯度下降更新参数
        w = w - learning_rate * dw
        b = b - learning_rate * db


    print("\n训练完成！")
    print(f"最终 w: {w:.4f}")
    print(f"最终 b: {b:.4f}")
    print(f"loss: {losses[epochs-1]:.4f}")

    # 5. 画拟合结果
    plt.figure()
    plt.scatter(X, y, label="data")
    plt.plot(X, w * X + b, label="model")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.show()

    # 6. 画 loss 曲线

    # 6.1 原始 loss 曲线
    plt.figure()
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss Curve")
    plt.show()

    # 6.2 前 50 轮 loss 曲线
    plt.figure()
    plt.plot(losses[:10])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss Curve - First 10 Epochs")
    plt.show()

    # 6.3 对数坐标 loss 曲线
    plt.figure()
    plt.plot(losses)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss Curve - Log Scale")
    plt.show()


if __name__ == "__main__":
    main()