"""使用 NumPy 手写支持 L2 正则化的二分类逻辑回归。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LogisticRegressionL2:
    """二分类逻辑回归。

    参数：
        learning_rate: 梯度下降学习率。
        l2_lambda: L2 正则化强度。0 表示不使用 L2。
        epochs: 训练轮数。
        record_every: 每隔多少轮记录一次 loss。
    """

    learning_rate: float = 0.01
    l2_lambda: float = 0.0
    epochs: int = 1000
    record_every: int = 50

    weights_: np.ndarray | None = field(default=None, init=False)
    bias_: float = field(default=0.0, init=False)
    history_: list[dict[str, float | int]] = field(
        default_factory=list,
        init=False,
    )

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate 必须大于 0。")
        if self.l2_lambda < 0:
            raise ValueError("l2_lambda 不能小于 0。")
        if self.epochs <= 0:
            raise ValueError("epochs 必须大于 0。")
        if self.record_every <= 0:
            raise ValueError("record_every 必须大于 0。")

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """数值稳定的 sigmoid。"""
        z = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def binary_cross_entropy(
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> float:
        """计算不含正则项的二元交叉熵。"""
        eps = 1e-12
        y_prob = np.clip(y_prob, eps, 1.0 - eps)

        return float(
            -np.mean(
                y_true * np.log(y_prob)
                + (1.0 - y_true) * np.log(1.0 - y_prob)
            )
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "LogisticRegressionL2":
        """使用批量梯度下降训练模型。"""
        X_train, y_train = self._validate_xy(X_train, y_train)

        if (X_val is None) != (y_val is None):
            raise ValueError("X_val 和 y_val 必须同时提供或同时省略。")

        if X_val is not None and y_val is not None:
            X_val, y_val = self._validate_xy(X_val, y_val)
            if X_val.shape[1] != X_train.shape[1]:
                raise ValueError("训练集和验证集的特征数量不一致。")

        n_samples, n_features = X_train.shape
        self.weights_ = np.zeros(n_features, dtype=float)
        self.bias_ = 0.0
        self.history_ = []

        for epoch in range(self.epochs):
            logits = X_train @ self.weights_ + self.bias_
            y_prob = self.sigmoid(logits)
            error = y_prob - y_train

            # 数据损失对参数的梯度。
            dw = X_train.T @ error / n_samples
            db = float(np.mean(error))

            # L2 只作用于 weights，不作用于 bias。
            dw += (self.l2_lambda / n_samples) * self.weights_

            self.weights_ -= self.learning_rate * dw
            self.bias_ -= self.learning_rate * db

            should_record = (
                epoch % self.record_every == 0
                or epoch == self.epochs - 1
            )
            if should_record:
                train_prob = self.predict_proba(X_train)
                train_bce = self.binary_cross_entropy(y_train, train_prob)
                regularization = (
                    self.l2_lambda
                    * float(np.sum(self.weights_ ** 2))
                    / (2.0 * n_samples)
                )

                record: dict[str, float | int] = {
                    "epoch": epoch,
                    "train_bce": train_bce,
                    "train_objective": train_bce + regularization,
                }

                if X_val is not None and y_val is not None:
                    val_prob = self.predict_proba(X_val)
                    record["val_bce"] = self.binary_cross_entropy(
                        y_val,
                        val_prob,
                    )

                self.history_.append(record)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """返回属于正类的概率。"""
        self._check_is_fitted()
        X = self._validate_X(X)

        if X.shape[1] != self.weights_.shape[0]:
            raise ValueError("输入特征数量与模型不一致。")

        return self.sigmoid(X @ self.weights_ + self.bias_)

    def predict(
        self,
        X: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """根据阈值返回 0/1 分类结果。"""
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold 必须位于 (0, 1) 内。")

        return (self.predict_proba(X) >= threshold).astype(int)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, float]:
        """返回 BCE、准确率和权重范数。"""
        X, y = self._validate_xy(X, y)
        y_prob = self.predict_proba(X)
        y_pred = (y_prob >= 0.5).astype(int)

        return {
            "loss": self.binary_cross_entropy(y, y_prob),
            "accuracy": float(np.mean(y_pred == y)),
            "weight_norm": float(np.linalg.norm(self.weights_)),
        }

    @staticmethod
    def _validate_X(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X 必须是二维数组。")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X 不能为空。")
        if not np.all(np.isfinite(X)):
            raise ValueError("X 中不能包含 NaN 或无穷大。")

        return X

    @classmethod
    def _validate_xy(
        cls,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        X = cls._validate_X(X)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X 和 y 的样本数量不一致。")
        if not np.all(np.isin(y, [0.0, 1.0])):
            raise ValueError("y 必须只包含 0 和 1。")

        return X, y

    def _check_is_fitted(self) -> None:
        if self.weights_ is None:
            raise RuntimeError("模型尚未 fit。")