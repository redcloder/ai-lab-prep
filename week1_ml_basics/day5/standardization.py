"""手写特征标准化工具。

1. 只能在训练集上调用 fit。
2. 验证集和测试集只能调用 transform。
3. 所有数据集必须使用训练集计算得到的同一组 mean_ 和 scale_。
"""

from __future__ import annotations

import numpy as np


class StandardScalerManual:
    """使用均值和标准差进行标准化：X_scaled = (X - mean) / std。"""

    def __init__(self, epsilon: float = 1e-12) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon 必须大于 0。")

        self.epsilon = epsilon
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScalerManual":
        """只使用传入的数据计算每个特征的均值和标准差。"""
        X = self._validate_X(X) #检查数据格式合法性

        self.mean_ = np.mean(X, axis=0)
        std = np.std(X, axis=0)

        # 常数特征的标准差为 0。将其缩放系数设为 1，避免除零。
        self.scale_ = np.where(std < self.epsilon, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """使用 fit 阶段保存的训练集参数转换数据。"""
        self._check_is_fitted()
        X = self._validate_X(X)

        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError(
                f"特征数量不一致：fit 时为 {self.mean_.shape[0]}，"
                f"当前数据为 {X.shape[1]}。"
            )

        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """先 fit 再 transform。通常只对训练集使用。"""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """将标准化后的数据还原到原始尺度。"""
        self._check_is_fitted()
        X_scaled = self._validate_X(X_scaled)

        if X_scaled.shape[1] != self.mean_.shape[0]:
            raise ValueError("特征数量与 scaler 不匹配。")

        return X_scaled * self.scale_ + self.mean_

    @staticmethod
    def _validate_X(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X 必须是二维数组，形状应为 (样本数, 特征数)。")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X 不能为空。")
        if not np.all(np.isfinite(X)):
            raise ValueError("X 中不能包含 NaN 或无穷大。")

        return X

    def _check_is_fitted(self) -> None:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScalerManual 尚未 fit。")


if __name__ == "__main__":
    X_train = np.array(
        [
            [18.0, 30000.0],
            [20.0, 35000.0],
            [22.0, 42000.0],
            [25.0, 50000.0],
        ]
    )
    X_test = np.array([[30.0, 80000.0]])

    scaler = StandardScalerManual()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("训练集均值：", scaler.mean_)
    print("训练集标准差：", scaler.scale_)
    print("标准化后的训练集：\n", X_train_scaled)
    print("使用训练集参数转换后的测试集：\n", X_test_scaled)