"""运行 Day 5 的特征标准化、学习率和 L2 对照实验。

运行方式：
    python experiments.py

程序会：
1. 生成一个具有不同特征尺度的二分类数据集；
2. 按 60% / 20% / 20% 划分训练集、验证集和测试集；
3. 只在训练集上拟合标准化参数；
4. 运行 Exp00～Exp03；
5. 自动生成 experiment_results.md。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from logistic_regression_l2 import LogisticRegressionL2
from standardization import StandardScalerManual


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    standardized: bool
    learning_rate: float
    l2_lambda: float
    description: str


def make_dataset(
    n_samples: int = 1200,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """生成三列尺度差异明显的二分类数据。"""
    rng = np.random.default_rng(random_state)

    # 先在相近尺度上生成潜在特征。
    latent_X = rng.normal(size=(n_samples, 3))

    # 标签由潜在特征决定，并加入少量噪声。
    logits = (
        1.8 * latent_X[:, 0]
        - 1.2 * latent_X[:, 1]
        + 0.9 * latent_X[:, 2]
        + rng.normal(scale=0.8, size=n_samples)
    )
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n_samples) < probabilities).astype(int)

    # 人为制造非常不同的特征尺度。
    X = latent_X.copy()
    X[:, 0] *= 1.0
    X[:, 1] *= 100.0
    X[:, 2] *= 0.01

    return X, y


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
) -> tuple[np.ndarray, ...]:
    """随机划分为 60% 训练集、20% 验证集、20% 测试集。"""
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(X))

    train_end = int(0.6 * len(X))
    val_end = int(0.8 * len(X))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return (
        X[train_idx],
        y[train_idx],
        X[val_idx],
        y[val_idx],
        X[test_idx],
        y[test_idx],
    )


def describe_loss_change(history: list[dict[str, float | int]]) -> str:
    """根据训练历史生成简短的 loss 变化描述。"""
    first = float(history[0]["train_bce"])
    last = float(history[-1]["train_bce"])
    losses = np.array([float(item["train_bce"]) for item in history])

    if not np.all(np.isfinite(losses)):
        return "出现非有限值，训练不稳定"

    increases = int(np.sum(np.diff(losses) > 1e-5))
    ratio = last / first if first > 0 else 1.0

    if ratio < 0.65 and increases == 0:
        return "稳定且明显下降"
    if ratio < 0.85 and increases <= 2:
        return "总体下降，较为稳定"
    if increases >= max(2, len(losses) // 4):
        return "存在明显震荡"
    if ratio >= 0.95:
        return "下降很慢或几乎不变"
    return "有所下降"


def run_experiments() -> list[dict[str, float | str | bool]]:
    X, y = make_dataset()
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = split_dataset(X, y)

    # 关键：scaler 只在训练集上 fit。
    scaler = StandardScalerManual()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    configs = [
        ExperimentConfig(
            name="Exp00",
            standardized=False,
            learning_rate=0.01,
            l2_lambda=0.0,
            description="未标准化基线",
        ),
        ExperimentConfig(
            name="Exp01",
            standardized=True,
            learning_rate=0.1,
            l2_lambda=0.0,
            description="较大学习率",
        ),
        ExperimentConfig(
            name="Exp02",
            standardized=True,
            learning_rate=0.01,
            l2_lambda=0.0,
            description="标准学习率基线",
        ),
        ExperimentConfig(
            name="Exp03",
            standardized=True,
            learning_rate=0.01,
            l2_lambda=0.1,
            description="加入 L2 正则化",
        ),
    ]

    results: list[dict[str, float | str | bool]] = []

    for config in configs:
        if config.standardized:
            train_X = X_train_scaled
            val_X = X_val_scaled
            test_X = X_test_scaled
        else:
            train_X = X_train
            val_X = X_val
            test_X = X_test

        model = LogisticRegressionL2(
            learning_rate=config.learning_rate,
            l2_lambda=config.l2_lambda,
            epochs=1500,
            record_every=50,
        )
        model.fit(train_X, y_train, val_X, y_val)

        train_metrics = model.evaluate(train_X, y_train)
        val_metrics = model.evaluate(val_X, y_val)
        test_metrics = model.evaluate(test_X, y_test)

        results.append(
            {
                "experiment": config.name,
                "standardized": config.standardized,
                "learning_rate": config.learning_rate,
                "l2_lambda": config.l2_lambda,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "test_loss": test_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "test_accuracy": test_metrics["accuracy"],
                "weight_norm": train_metrics["weight_norm"],
                "loss_change": describe_loss_change(model.history_),
                "description": config.description,
            }
        )

    return results


def build_markdown(
    results: list[dict[str, float | str | bool]],
) -> str:
    """将实验结果转换为 Markdown。"""
    lines = [
        "# Day 5 实验结果",
        "",
        "## 实验设置",
        "",
        "- 数据集：程序自动生成的二分类数据，共 1200 个样本、3 个特征。",
        "- 原始特征尺度约为 `1`、`100` 和 `0.01`。",
        "- 数据划分：训练集 60%，验证集 20%，测试集 20%。",
        "- 标准化参数只在训练集上计算。",
        "- 所有实验训练 1500 轮。",
        "",
        "## 实验结果",
        "",
        "| 实验 | 标准化 | 学习率 | L2 | Train Loss | Val Loss | "
        "Train Acc | Val Acc | Test Acc | 权重范数 | Loss 变化 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for item in results:
        lines.append(
            "| {experiment} | {standardized} | {learning_rate:.3g} | "
            "{l2_lambda:.3g} | {train_loss:.4f} | {val_loss:.4f} | "
            "{train_accuracy:.4f} | {val_accuracy:.4f} | "
            "{test_accuracy:.4f} | {weight_norm:.4f} | {loss_change} |".format(
                experiment=item["experiment"],
                standardized="是" if item["standardized"] else "否",
                learning_rate=float(item["learning_rate"]),
                l2_lambda=float(item["l2_lambda"]),
                train_loss=float(item["train_loss"]),
                val_loss=float(item["val_loss"]),
                train_accuracy=float(item["train_accuracy"]),
                val_accuracy=float(item["val_accuracy"]),
                test_accuracy=float(item["test_accuracy"]),
                weight_norm=float(item["weight_norm"]),
                loss_change=item["loss_change"],
            )
        )

    result_by_name = {
        str(item["experiment"]): item
        for item in results
    }

    exp00 = result_by_name["Exp00"]
    exp01 = result_by_name["Exp01"]
    exp02 = result_by_name["Exp02"]
    exp03 = result_by_name["Exp03"]

    lines.extend(
        [
            "",
            "## 现象分析",
            "",
            "### 1. 标准化的影响：Exp00 对比 Exp02",
            "",
            f"- Exp00 的验证集 loss 为 `{float(exp00['val_loss']):.4f}`，"
            f"Exp02 为 `{float(exp02['val_loss']):.4f}`。",
            f"- Exp00 的 loss 变化表现为“{exp00['loss_change']}”，"
            f"Exp02 为“{exp02['loss_change']}”。",
            "- 未标准化时，大尺度特征会产生更大的梯度，"
            "同一个学习率难以同时适应所有特征。",
            "",
            "### 2. 学习率的影响：Exp01 对比 Exp02",
            "",
            f"- Exp01 使用学习率 `0.1`，验证集准确率为 "
            f"`{float(exp01['val_accuracy']):.4f}`。",
            f"- Exp02 使用学习率 `0.01`，验证集准确率为 "
            f"`{float(exp02['val_accuracy']):.4f}`。",
            "- 较大的学习率通常前期下降更快，但过大时可能震荡；"
            "较小的学习率更稳，但需要更多训练轮数。当然，这里训练数据较为简单，故不明显。",
            "",
            "### 3. L2 的影响：Exp02 对比 Exp03",
            "",
            f"- 不使用 L2 时权重范数为 "
            f"`{float(exp02['weight_norm']):.4f}`。",
            f"- 使用 L2 后权重范数为 "
            f"`{float(exp03['weight_norm']):.4f}`。",
            "- L2 会惩罚较大的权重。是否提高验证集准确率取决于"
            "数据、正则化强度以及模型是否真的发生过拟合。",
            "",
            "## 今日结论",
            "",
            "1. 标准化参数必须只在训练集上拟合。",
            "2. 特征尺度会直接影响梯度大小和学习率选择。",
            "3. 学习率决定每次参数更新的步长。",
            "4. L2 通过限制权重大小降低模型复杂度。",
            "5. 判断模型效果不能只看训练集，还要比较验证集和测试集。",
            "",
            "> 说明：这些数值来自固定随机种子生成的数据。"
            "修改数据、学习率、L2 强度或训练轮数后，结果会变化。",
        ]
    )

    return "\n".join(lines) + "\n"


def print_results(results: list[dict[str, float | str | bool]]) -> None:
    for item in results:
        print(
            f"{item['experiment']}: "
            f"standardized={item['standardized']}, "
            f"lr={item['learning_rate']}, "
            f"l2={item['l2_lambda']}, "
            f"train_loss={float(item['train_loss']):.4f}, "
            f"val_loss={float(item['val_loss']):.4f}, "
            f"val_acc={float(item['val_accuracy']):.4f}, "
            f"test_acc={float(item['test_accuracy']):.4f}, "
            f"weight_norm={float(item['weight_norm']):.4f}"
        )


def main() -> None:
    results = run_experiments()
    print_results(results)

    output_path = Path(__file__).with_name("experiment_results.md")
    output_path.write_text(
        build_markdown(results),
        encoding="utf-8",
    )
    print(f"\n实验结果已写入：{output_path}")


if __name__ == "__main__":
    main()