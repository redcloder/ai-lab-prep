import numpy as np

np.random.seed(42)

# m 个样本，n 个特征
m, n = 100, 3

X = np.random.randn(m, n)
true_w = np.array([[2.0], [-3.0], [1.5]])
true_b = 4.0

y = X @ true_w + true_b + 0.1 * np.random.randn(m, 1)

w = np.zeros((n, 1))
b = 0.0
lr = 0.05
epochs = 1000

for epoch in range(epochs):
    y_pred = X @ w + b

    loss = np.mean((y_pred - y) ** 2)

    dw = (2 / m) * X.T @ (y_pred - y)
    db = (2 / m) * np.sum(y_pred - y)

    w -= lr * dw
    b -= lr * db

    if epoch % 100 == 0:
        print(f"epoch {epoch}, loss = {loss:.4f}")

print("learned w:")
print(w)
print("learned b:", b)