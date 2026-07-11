import numpy as np
import matplotlib.pyplot as plt


def split_data(X, y, train_ratio=0.6, val_ratio=0.2, seed=42):
    """随机打乱后，将数据划分为 train / val / test。"""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))

    train_end = int(len(X) * train_ratio)
    val_end = train_end + int(len(X) * val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return (
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
        X[test_idx], y[test_idx],
    )


def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


def main():
    # 1. 生成数据：y = 3x + 2 + 噪声
    rng = np.random.default_rng(42)
    X = np.linspace(0, 10, 100)
    noise = rng.normal(0, 2, size=100)
    y = 3 * X + 2 + noise

    # 2. 数据划分：60% train，20% val，20% test
    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
    ) = split_data(X, y)

    print(
        f"train: {len(X_train)}, "
        f"val: {len(X_val)}, "
        f"test: {len(X_test)}"
    )

    # 3. 只使用训练集更新参数
    w = 0.0
    b = 0.0
    learning_rate = 0.003
    epochs = 4000
    n_train = len(X_train)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # forward：训练集
        train_pred = w * X_train + b
        train_loss = mse(train_pred, y_train)

        # validation：验证集只评估，不参与梯度计算
        val_pred = w * X_val + b
        val_loss = mse(val_pred, y_val)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # backward：梯度只能来自训练集
        error = train_pred - y_train
        dw = (2 / n_train) * np.sum(error * X_train)
        db = (2 / n_train) * np.sum(error)

        w -= learning_rate * dw
        b -= learning_rate * db

        if epoch % 500 == 0:
            print(
                f"epoch {epoch:4d} | "
                f"train_mse={train_loss:.4f} | "
                f"val_mse={val_loss:.4f}"
            )

    # 4. 训练结束后，测试集只做最终评估
    final_train_mse = mse(w * X_train + b, y_train)
    final_val_mse = mse(w * X_val + b, y_val)
    final_test_mse = mse(w * X_test + b, y_test)

    print("\n训练完成")
    print(f"w = {w:.4f}, b = {b:.4f}")
    print(f"train MSE = {final_train_mse:.4f}")
    print(f"val   MSE = {final_val_mse:.4f}")
    print(f"test  MSE = {final_test_mse:.4f}")

    # 5. 拟合效果
    x_line = np.linspace(X.min(), X.max(), 200)

    plt.figure()
    plt.scatter(X_train, y_train, label="train")
    plt.scatter(X_val, y_val, label="val")
    plt.scatter(X_test, y_test, label="test")
    plt.plot(x_line, w * x_line + b, label="model")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression: Train / Val / Test")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 6. train / val loss 曲线
    plt.figure()
    plt.plot(train_losses[:10], label="train loss")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title("Linear Regression Loss Curves-First 10 train epochs")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    plt.plot(val_losses[:10], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title("Linear Regression Loss Curves-First 10 val epochs")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()