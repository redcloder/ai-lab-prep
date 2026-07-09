import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


np.random.seed(42)

# 生成一个简单二分类数据集
m = 200
n = 2

X = np.random.randn(m, n)

true_w = np.array([[2.0], [-3.0]])
true_b = 0.5

z = X @ true_w + true_b
prob = sigmoid(z)

# 根据概率生成 0/1 标签
y = (prob >= 0.5).astype(int)

# 初始化参数
w = np.zeros((n, 1))
b = 0.0

lr = 0.1
epochs = 1000

for epoch in range(epochs):
    # 模型预测
    z = X @ w + b
    y_pred = sigmoid(z)

    # 计算交叉熵损失
    eps = 1e-8
    loss = -np.mean(
        y * np.log(y_pred + eps) +
        (1 - y) * np.log(1 - y_pred + eps)
    )

    # backward
    dz = y_pred - y
    dw = (1 / m) * X.T @ dz
    db = (1 / m) * np.sum(dz)

    # 更新
    w -= lr * dw
    b -= lr * db

    if epoch % 100 == 0:
        predictions = (y_pred >= 0.5).astype(int)
        accuracy = np.mean(predictions == y)
        print(f"epoch {epoch}, loss = {loss:.4f}, acc = {accuracy:.4f}")

print("learned w:")
print(w)
print("learned b:", b)

final_pred = (sigmoid(X @ w + b) >= 0.5).astype(int)
final_acc = np.mean(final_pred == y)

print("final accuracy:", final_acc)