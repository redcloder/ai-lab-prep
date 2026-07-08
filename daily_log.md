# Daily Log

## Day 1：环境与仓库

日期：2026-07-07

### 今日任务

- [x] 完成一道数组基础算法题：两数之和
- [x] 复习 Python 基础语法
- [x] 复习 NumPy 数组、矩阵维度、广播
- [x] 创建 ai-lab-prep 仓库
- [x] 配置 Conda 环境和 PyCharm
- [x] 提交 README.md、requirements.txt、daily_log.md

### 今日学习记录

#### 1. Python / NumPy

今天复习了 NumPy 中数组的创建、shape、维度变化和广播机制。

#### 2. 算法题

完成题目：两数之和。

思路：

- 使用哈希表记录已经遍历过的数字
- 对每个数字 x，检查 target - x 是否已经出现
- 如果出现，返回两个下标

#### 3. 遇到的问题

- 暂无

#### 4. 明日计划

- 继续 NumPy 基础
- 学习矩阵运算
- 完成一个小型数据处理练习

## Day 2：线性回归与 NumPy 实践

日期：2026-07-08

### 今日任务

- [x] 完成一道数组 / 哈希题：存在重复元素
- [x] 学习线性回归基本形式 `y = wx + b`
- [x] 学习 MSE loss
- [x] 学习梯度下降更新参数
- [x] 用 NumPy 手写线性回归
- [x] 画出拟合结果图和 loss 曲线
- [ ] 系统学习矩阵乘法和矩阵维度规则

### 今日学习记录

#### 1. 算法题

完成题目：存在重复元素。

思路：

- 使用 `set` 或 `dict` 记录已经出现过的数字
- 遍历数组时，判断当前数字是否已经出现过
- 如果已经出现过，说明存在重复元素，返回 `True`
- 如果遍历结束都没有发现重复元素，返回 `False`

代码示例：

```python
from typing import List


def contains(nums: List[int]) -> bool:
    seen = {}

    for i, num in enumerate(nums):
        if num in seen:
            return True

        seen[num] = i

    return False
```

复杂度分析：

- 时间复杂度：`O(n)`
- 空间复杂度：`O(n)`

#### 2. NumPy / 线性回归

今天使用 NumPy 生成了一组简单的线性数据。

真实规律为：

```python
y = 3 * X + 2 + noise
```

其中：

- `X` 是输入数据
- `y` 是真实标签
- `noise` 是随机噪声
- 真实的线性关系大致是 `y = 3x + 2`

相关 NumPy 函数：

- `np.random.seed(42)`：固定随机种子，使每次运行结果一致
- `np.linspace(0, 10, 100)`：在 0 到 10 之间均匀生成 100 个数
- `np.random.randn(100)`：生成 100 个服从标准正态分布的随机数
- `np.mean(...)`：求平均值
- `np.sum(...)`：求和

#### 3. 线性回归模型

今天实现的线性回归模型为：

```python
y_pred = w * X + b
```

其中：

- `X`：输入数据
- `y_pred`：模型预测值
- `w`：权重，也可以理解为直线的斜率
- `b`：偏置，也可以理解为直线的截距

模型训练的目标是让预测值 `y_pred` 尽可能接近真实值 `y`。

#### 4. MSE Loss

今天使用 MSE 作为损失函数。

代码为：

```python
loss = np.mean((y_pred - y) ** 2)
```

MSE 的含义是：预测值和真实值之间误差平方的平均值。

使用 MSE 的原因：

- 平方可以避免正负误差相互抵消
- 平方会放大较大的错误，使模型更加重视大误差
- MSE 适合用于线性回归这类回归任务

#### 5. 梯度下降

今天使用梯度下降更新参数 `w` 和 `b`。

梯度计算代码：

```python
dw = (2 / n) * np.sum((y_pred - y) * X)
db = (2 / n) * np.sum(y_pred - y)
```

参数更新代码：

```python
w = w - learning_rate * dw
b = b - learning_rate * db
```

理解：

- `dw` 表示 loss 对 `w` 的变化方向
- `db` 表示 loss 对 `b` 的变化方向
- 梯度下降每一步都在更新 `w` 和 `b`
- 更新方向是让 loss 下降的方向
- `learning_rate` 控制每次参数更新的步长

#### 6. 画图与观察

今天使用 `matplotlib.pyplot` 画了两类图：

拟合结果图：

```python
plt.scatter(X, y, label="data")
plt.plot(X, w * X + b, label="model")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression Fit")
plt.legend()
plt.show()
```

Loss 曲线：

```python
plt.plot(losses)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Loss Curve")
plt.show()
```

观察结果：

- 拟合直线大致穿过散点中心
- loss 在前几轮下降非常快
- 原始 loss 曲线不明显，是因为第 0 轮 loss 太大，把 y 轴拉得很高
- 使用 log 坐标后可以看出 loss 确实快速下降
- 由于数据中加入了随机噪声，最终 loss 不会下降到 0

#### 7. 遇到的问题

1. Loss 曲线看起来不明显。

原因：

- 第 0 轮 loss 非常大
- 前几轮 loss 下降太快
- 后面的 loss 变化被压在图像底部，看起来像一条平线

解决方法：

- 只画前十轮的 loss
- 使用 `plt.yscale("log")` 设置对数坐标

2. 发现当前例子中 loss 下降过快。

原因：

- 这个任务比较简单
- 数据本身就是线性关系
- 模型形式和数据真实规律一致

把学习率降低后观察到的现象可更明显，不过需增加epochs

3. 矩阵运算还没有系统学习。

今天主要练习的是 NumPy 数组向量化运算，还没有系统学习矩阵乘法、矩阵维度规则、转置等内容。

#### 8. 今日总结

今天基本完成了 Day 2 的任务。

今天能够解释：

- `y = wx + b` 是一个线性模型
- `w` 控制直线的斜率
- `b` 控制直线的上下平移
- MSE 用来衡量预测值和真实值之间的平均平方误差
- 梯度下降每一步都在更新 `w` 和 `b`
- loss 下降说明模型预测正在变好
- loss 不一定会下降到 0，因为数据中可能存在噪声

今天的主要产出：

- 完成算法题：存在重复元素
- 完成 `day2_linear_regression_numpy.py`
- 画出线性回归拟合图
- 画出 loss 曲线
- 理解了 NumPy 向量化计算在线性回归中的作用

#### 9. 明日计划

- 系统学习 NumPy 矩阵运算
- 理解矩阵乘法的维度规则
- 学习 `reshape`、转置 `.T`、矩阵乘法 `@`
- 学习多特征线性回归
- 尝试把单变量线性回归改成多变量线性回归
- 逻辑回归

## Day2.5:矩阵运算
日期：2026-07-08
### 任务
- [x] 系统学习矩阵乘法和矩阵维度规则
### 学习记录
- 想了想还是今天把矩阵运算搞定了，为明天多特征线性回归和逻辑回归做准备 
- 从 shape,reshape,@,*几个角度进行矩阵复习，还有多特征求值
