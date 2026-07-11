import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    # clip 可避免 exp 溢出
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy(y_prob, y_true):
    eps = 1e-8
    return -np.mean(
        y_true * np.log(y_prob + eps)
        + (1 - y_true) * np.log(1 - y_prob + eps)
    )


def accuracy(y_prob, y_true, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return np.mean(y_pred == y_true)


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


def evaluate(X, y, w, b):
    y_prob = sigmoid(X @ w + b)
    return binary_cross_entropy(y_prob, y), accuracy(y_prob, y)


def main():
    # 1. 生成二分类数据
    rng = np.random.default_rng(42)
    m = 300
    n = 2

    X = rng.normal(size=(m, n))
    true_w = np.array([[2.0], [-3.0]])
    true_b = 0.5

    prob = sigmoid(X @ true_w + true_b)

    # 按概率采样标签，比直接 prob >= 0.5 更接近真实带噪声数据
    y = rng.binomial(1, prob).reshape(-1, 1)

    # 2. 数据划分
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

    # 3. 初始化参数
    w = np.zeros((n, 1))
    b = 0.0

    learning_rate = 0.1
    epochs = 1000
    n_train = len(X_train)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # 4. 只使用训练集训练
    for epoch in range(epochs):
        # 模型训练
        train_prob = sigmoid(X_train @ w + b)
        train_loss = binary_cross_entropy(train_prob, y_train)
        train_acc = accuracy(train_prob, y_train)

        # validation：只评估，不更新参数
        val_prob = sigmoid(X_val @ w + b)
        val_loss = binary_cross_entropy(val_prob, y_val)
        val_acc = accuracy(val_prob, y_val)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # backward：梯度只由训练集计算
        dz = train_prob - y_train
        dw = (1 / n_train) * X_train.T @ dz
        db = (1 / n_train) * np.sum(dz)

        w -= learning_rate * dw
        b -= learning_rate * db

        if epoch % 100 == 0:
            print(
                f"epoch {epoch:4d} | "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f} | "
                f"train_acc={train_acc:.4f}, "
                f"val_acc={val_acc:.4f}"
            )

    # 5. 最终评估：测试集不要参与调参
    train_loss, train_acc = evaluate(X_train, y_train, w, b)
    val_loss, val_acc = evaluate(X_val, y_val, w, b)
    test_loss, test_acc = evaluate(X_test, y_test, w, b)

    print("\n训练完成")
    print("learned w:")
    print(w)
    print(f"learned b: {b:.4f}")
    print(
        f"train | loss={train_loss:.4f}, accuracy={train_acc:.4f}"
    )
    print(
        f"val   | loss={val_loss:.4f}, accuracy={val_acc:.4f}"
    )
    print(
        f"test  | loss={test_loss:.4f}, accuracy={test_acc:.4f}"
    )

    # 6. loss 曲线
    plt.figure()
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("binary cross-entropy")
    plt.title("Logistic Regression Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 7. accuracy 曲线
    plt.figure()
    plt.plot(train_accuracies, label="train accuracy")
    plt.plot(val_accuracies, label="val accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Logistic Regression Accuracy Curves")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()