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

## Day 3：逻辑回归

日期：2026-07-09

### 今日任务

-[x] 学习多特征线性回归
-[x] 将单变量线性回归改成多变量线性回归
-[x] 完成一道字符串 / 哈希算法题：有效的字母异位词
-[x] 学习 sigmoid 函数
-[x] 理解二分类任务
-[x] 学习交叉熵损失函数
-[x] 使用 NumPy 手写逻辑回归
-[x] 完成 `logistic_regression_numpy.py`

### 今日学习记录

#### 1. 多特征线性回归

今天学习了多特征线性回归的基本形式：

```python
y_pred = X @ w + b
```

其中 `X` 表示样本特征矩阵，`w` 表示权重向量，`b` 表示偏置项。

相比单变量线性回归，多特征线性回归的核心变化是从：

```python
y_pred = w * x + b
```

变成：

```python
y_pred = X @ w + b
```

重点理解了矩阵维度：

```python
X.shape = (m, n)
w.shape = (n, 1)
y_pred.shape = (m, 1)
```

还学习了梯度计算公式：

```python
dw = (2 / m) * X.T @ (y_pred - y)
db = (2 / m) * np.sum(y_pred - y)
```

其中 `dw` 表示每个特征对应权重的梯度，`db` 表示整体预测偏高或偏低时对偏置项的调整方向。

#### 2. 算法题

完成题目：有效的字母异位词。

思路：

* 先判断两个字符串长度是否相同
* 使用哈希表统计第一个字符串中每个字符出现的次数
* 遍历第二个字符串，用其中的字符去抵消哈希表中的计数
* 如果遇到不存在的字符，返回 `False`
* 如果某个字符计数被减成负数，返回 `False`
* 如果全部抵消成功，返回 `True`

核心代码：

```python
count[ch] = count.get(ch, 0) + 1
count[ch] -= 1
```

这个题加深了对哈希表计数问题的理解。

#### 3. 逻辑回归

今天学习了逻辑回归的基本流程：

```python
z = X @ w + b
y_pred = sigmoid(z)
```

其中 sigmoid 函数可以把任意实数压缩到 0 到 1 之间：

```python
sigmoid(z) = 1 / (1 + exp(-z))
```

因此逻辑回归的输出可以理解为样本属于正类的概率。

如果输出结果大于等于 0.5，通常预测为 1；否则预测为 0。

#### 4. 交叉熵损失

今天学习了二分类交叉熵损失：

```python
loss = -mean(y * log(y_pred) + (1 - y) * log(1 - y_pred))
```

交叉熵主要惩罚模型的错误预测，尤其是模型非常自信但预测错误的情况。

例如：

* 真实标签是 1，但模型预测为 0.01，loss 会很大
* 真实标签是 0，但模型预测为 0.99，loss 会很大
* 模型预测正确且越自信，loss 越小

#### 5. 今日代码

完成文件：

```text
logistic_regression_numpy.py
```

实现内容包括：

* 生成简单二分类数据集
* 定义 sigmoid 函数
* 初始化参数 `w` 和 `b`
* 使用交叉熵作为 loss
* 使用梯度下降更新参数
* 输出训练过程中的 loss 和 accuracy

### 遇到的问题

* 对多特征线性回归中 `X.T @ (y_pred - y)` 的含义一开始不够清楚，后来理解为：把每个样本的误差按照不同特征汇总，得到每个权重对应的梯度。
* 对逻辑回归和线性回归的区别需要继续巩固，目前理解为：逻辑回归是在 `X @ w + b` 后面加了 sigmoid，并使用交叉熵损失处理二分类问题。

### 明日计划

* 学习训练集、验证集、测试集的划分
* 理解过拟合和泛化能力
* 完成一道双指针算法题
* 给前两天的模型加入数据划分和评估
* 记录训练集和测试集上的 accuracy、loss
* 整理一份简单实验记录

## Day 4：训练集、验证集、测试集

日期：2026-07-11

### 今日任务

* [x] 完成一道双指针算法题：两数之和 II
* [x] 理解训练集、验证集和测试集的作用
* [x] 理解过拟合与泛化
* [x] 为线性回归模型添加数据划分和评估
* [x] 为逻辑回归模型添加数据划分和评估
* [x] 记录训练集和验证集的 loss
* [x] 记录逻辑回归的 accuracy
* [x] 绘制 loss 和 accuracy 曲线
* [x] 学习 NumPy 随机数生成器 `default_rng`

### 今日学习记录

#### 1. 训练集、验证集和测试集

今天学习了机器学习中常见的数据划分方式。

* 训练集用于计算预测结果、损失函数和梯度，并更新模型参数
* 验证集用于观察模型在未参与训练的数据上的表现，并辅助选择学习率、训练轮数等超参数
* 测试集用于模型训练和调参完成后的最终评估

本次实验将数据按照以下比例划分：

* 训练集：60%
* 验证集：20%
* 测试集：20%

数据划分前需要先随机打乱样本下标，并保证 `X` 和 `y` 使用相同的下标进行划分，避免特征和标签错位。

#### 2. 过拟合与泛化

泛化能力表示模型在没有见过的数据上的表现。

如果模型在训练集上的效果很好，但在验证集和测试集上的效果明显较差，说明模型可能出现了过拟合。

常见的过拟合现象：

* 训练集 loss 持续下降
* 验证集 loss 先下降，之后开始上升
* 训练集 accuracy 持续上升
* 验证集 accuracy 不再上升，甚至开始下降

因此不能只关注训练集指标，还需要同时观察验证集指标。

#### 3. 线性回归评估

为之前的线性回归模型添加了训练集、验证集和测试集划分。

训练时只使用训练集计算梯度：

```python
train_pred = w * X_train + b

error = train_pred - y_train
dw = (2 / len(X_train)) * np.sum(error * X_train)
db = (2 / len(X_train)) * np.sum(error)
```

验证集只用于计算 loss，不参与参数更新。

线性回归属于回归问题，因此不使用 accuracy，而是使用 MSE 进行评估。

```python
mse = np.mean((y_pred - y_true) ** 2)
```

训练完成后分别计算：

* Train MSE
* Validation MSE
* Test MSE

并绘制训练集和验证集的 MSE 曲线。

#### 4. 逻辑回归评估

为之前的逻辑回归模型添加了训练集、验证集和测试集划分。

训练过程中记录了：

* Train loss
* Validation loss
* Train accuracy
* Validation accuracy

逻辑回归使用二元交叉熵作为损失函数：

```python
loss = -np.mean(
    y * np.log(y_pred + eps)
    + (1 - y) * np.log(1 - y_pred + eps)
)
```

使用 `0.5` 作为分类阈值：

```python
predictions = (y_pred >= 0.5).astype(int)
accuracy = np.mean(predictions == y)
```

训练结束后，使用测试集计算最终的 test loss 和 test accuracy。

#### 5. NumPy 随机数生成器

今天学习了 NumPy 推荐的随机数生成方式：

```python
rng = np.random.default_rng(42)
```

其中 `42` 是随机种子。相同的随机种子和相同的调用顺序可以产生相同的随机结果，方便复现实验。

使用到的函数：

```python
rng.normal(...)
```

用于生成正态分布随机数，可以用于生成特征或噪声。

```python
rng.permutation(...)
```

用于生成随机排列，可以用于打乱样本下标。

```python
rng.binomial(1, prob)
```

用于根据概率生成 `0` 或 `1` 标签，使分类数据带有一定随机噪声。

#### 6. 算法题

完成题目：两数之和 II——输入有序数组。

思路：

* 数组已经按照从小到大排列
* 使用 `left` 指向数组左端
* 使用 `right` 指向数组右端
* 计算两个指针所指数字之和
* 如果和等于 `target`，返回两个下标
* 如果和小于 `target`，说明需要增大当前和，因此右移左指针
* 如果和大于 `target`，说明需要减小当前和，因此左移右指针

核心代码：

```python
left = 0
right = len(numbers) - 1

while left < right:
    current_sum = numbers[left] + numbers[right]

    if current_sum == target:
        return [left + 1, right + 1]
    elif current_sum < target:
        left += 1
    else:
        right -= 1
```

时间复杂度：

```text
O(n)
```

空间复杂度：

```text
O(1)
```

双指针能够成立的关键是数组已经有序，每次移动指针时都可以排除一个不可能参与答案的元素。

#### 7. 遇到的问题

* 线性回归不能使用 accuracy 作为主要评价指标
* LeetCode 167 返回的是从 1 开始的下标，而 Python 列表下标从 0 开始
* 使用随机数生成器时，同一个 `rng` 连续调用会生成不同结果，但完整随机序列可以通过固定 seed 复现

#### 8. 今日总结

今天将前两天的线性回归和逻辑回归代码改造成了更完整的机器学习训练流程。

相比之前直接在全部数据上训练和评估，今天实现了训练集、验证集和测试集的分离，并学会了通过 loss 和 accuracy 曲线观察模型的训练过程。

同时理解了过拟合和泛化的基本概念，知道了模型不能只关注训练集表现，还需要关注未参与训练的数据上的表现。

算法部分学习了有序数组中的双指针方法，理解了如何根据当前两数之和与目标值的关系移动左右指针。

#### 9. 明日计划

* 学习特征缩放和标准化
* 理解不同特征尺度对梯度下降的影响
* 手写均值和标准差标准化
* 学习L2正则化和过拟合

## Day 5：特征标准化、学习率与 L2 正则化

日期：2026-07-13

### 今日任务

* [x] 完成一道滑动窗口简单题
* [x] 学习特征缩放和标准化
* [x] 理解不同特征尺度对梯度下降的影响
* [x] 手写均值和标准差标准化
* [x] 理解标准化过程中的数据泄漏问题
* [x] 学习不同学习率对模型训练的影响
* [x] 学习过拟合和 L2 正则化
* [x] 为逻辑回归加入 L2 正则化
* [x] 完成不同学习率、标准化和 L2 的对照实验
* [x] 生成并整理实验结果表

### 今日学习记录

#### 1. 特征标准化

今天学习了使用均值和标准差进行特征标准化。

标准化公式：
$$
x'=\frac{x-\mu}{\sigma}
$$

标准化后，每个特征通常具有：

* 均值接近 0
* 标准差接近 1
* 不同特征处于相近的数值尺度

当不同特征的尺度差异很大时，大尺度特征通常会产生更大的梯度，使梯度下降难以为所有特征选择合适的学习率，可能出现收敛缓慢或 loss 震荡的问题。

今天手写了 `StandardScalerManual`，实现了：

* `fit`
* `transform`
* `fit_transform`
* `inverse_transform`

同时进一步理解了 NumPy 数组的 `shape`：

* `X.shape[0]` 表示样本数量
* `X.shape[1]` 表示特征数量
* `mean_.shape[0]` 表示训练时记录的特征数量

因此下面的代码用于检查当前数据的特征数量是否与标准化器训练时一致：

```python
if X.shape[1] != self.mean_.shape[0]:
    raise ValueError("特征数量不一致")
```

#### 2. 数据泄漏

今天理解了为什么不能先对全部数据进行标准化，再划分训练集、验证集和测试集。

均值和标准差也是从数据中学习得到的参数。如果使用全部数据计算它们，那么验证集和测试集的分布信息会提前进入训练过程，造成数据泄漏。

正确流程是：

1. 先划分训练集、验证集和测试集
2. 只在训练集上计算均值和标准差
3. 使用训练集得到的同一组参数转换所有数据集

代码流程：

```python
scaler = StandardScalerManual()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

可以总结为：

> `fit` 只能作用于训练集，`transform` 可以作用于训练集、验证集和测试集。

#### 3. 学习率

学习率决定每次梯度下降更新参数的步长：

```python
w -= learning_rate * dw
b -= learning_rate * db
```

不同学习率的表现：

* 学习率过大时，loss 可能震荡、上升，甚至出现 `nan`
* 学习率过小时，loss 下降较慢，需要更多训练轮数
* 学习率合适时，loss 能够较快且稳定地下降

今天通过实验比较了学习率 `0.1` 和 `0.01` 的训练结果。

#### 4. L2 正则化

今天为手写逻辑回归加入了 L2 正则化。

加入正则化后的目标函数为：

$$
J(w,b)= L_{\text{BCE}}+ \frac{\lambda}{2n}\sum_jw_j^2
$$

对应的权重梯度为：

$$
dw
= \frac{1}{n}X^T(p-y)+ \frac{\lambda}{n}w
$$

代码实现：

```python
dw += (self.l2_lambda / n_samples) * self.weights_
```

L2 会惩罚过大的权重，使参数更新时产生权重衰减，从而限制模型复杂度，缓解过拟合。

通常不对偏置 `b` 使用 L2 正则化，因为偏置主要控制决策边界的整体平移，不直接表示某个特征的重要程度。

#### 5. 手写逻辑回归训练过程

今天进一步拆解并理解了逻辑回归的完整训练流程：

```python
logits = X_train @ self.weights_ + self.bias_
y_prob = self.sigmoid(logits)
error = y_prob - y_train

dw = X_train.T @ error / n_samples
db = np.mean(error)

dw += (self.l2_lambda / n_samples) * self.weights_

self.weights_ -= self.learning_rate * dw
self.bias_ -= self.learning_rate * db
```

各变量的含义：

* `logits`：线性模型输出
* `y_prob`：经过 sigmoid 后得到的正类概率
* `error`：预测概率与真实标签之差
* `dw`：权重梯度
* `db`：偏置梯度

同时理解了训练集 BCE 和训练目标函数的区别：

* `train_bce` 只表示模型的预测误差
* `train_objective` 包含 BCE 和 L2 正则项

#### 6. 实验结果

今天完成了四组对照实验：

| 实验    | 标准化 |  学习率 |  L2 | Train Loss | Val Loss | Val Acc |   权重范数 |
| ----- | --- | ---: | --: | ---------: | -------: | ------: | -----: |
| Exp00 | 否   | 0.01 |   0 |     4.2198 |   4.6984 |  0.6833 | 2.6917 |
| Exp01 | 是   |  0.1 |   0 |     0.4240 |   0.5199 |  0.7458 | 2.3392 |
| Exp02 | 是   | 0.01 |   0 |     0.4366 |   0.5026 |  0.7500 | 1.6650 |
| Exp03 | 是   | 0.01 | 0.1 |     0.4367 |   0.5026 |  0.7500 | 1.6636 |

实验现象：

* Exp00 未进行标准化，loss 较高且训练不稳定
* 标准化后，模型的 loss 明显下降，验证集准确率有所提高
* 学习率 `0.1` 收敛速度较快，但较大的学习率更容易产生震荡
* 学习率 `0.01` 训练更加稳定
* 加入 L2 后，权重范数从 `1.6650` 降低到 `1.6636`
* 本次数据中 Exp02 和 Exp03 的验证集表现基本相同，说明当前数据没有出现明显过拟合，或者 L2 强度较弱

#### 7. Loss 趋势判断

今天学习了根据训练历史自动描述 loss 变化趋势的函数。

主要使用了两个指标：

```python
ratio = last / first
```

用于判断最终 loss 相对于初始 loss 下降了多少。

```python
increases = int(np.sum(np.diff(losses) > 1e-5))
```

用于统计相邻记录中 loss 明显上升的次数。

其中：

* `np.diff(losses)` 计算相邻 loss 的差
* 差值大于 `1e-5` 表示发生一次明显上升
* `np.sum` 统计上升次数
* `ratio` 判断整体下降幅度
* `increases` 判断训练过程是否稳定

该函数属于启发式判断，使用的阈值并不是严格的数学标准，主要用于自动填写实验现象。

### 遇到的问题

#### 1. 不理解 `shape[0]` 和 `shape[1]`

解决：

* 二维特征矩阵的形状为 `(样本数, 特征数)`
* `shape[0]` 是样本数
* `shape[1]` 是特征数
* 一维均值数组的形状为 `(特征数,)`

#### 2. 不理解为什么标准化必须先划分数据

解决：

* 均值和标准差也是从数据中学习的参数
* 使用验证集或测试集计算这些参数会造成数据泄漏
* 所有预处理参数只能在训练集上拟合

#### 3. 不理解 L2 如何影响参数更新

解决：

L2 在原始梯度上增加：

```python
(self.l2_lambda / n_samples) * self.weights_
```

使每次更新都倾向于缩小权重，从而限制模型复杂度。

### 明日计划

* 完成一道栈或队列简单题
* 学习 PyTorch Tensor 的创建和基本操作
* 复习 Tensor 的 `shape` 和维度变化
* 理解 Tensor 和 NumPy array 的区别
* 学习 `requires_grad` 的作用
* 理解计算图和自动求导
* 理解 `loss.backward()` 执行了什么
* 使用 PyTorch 重写线性回归

## Day 6：Tensor 与 autograd

日期：2026-07-14

### 今日任务

* [ ] 完成一道栈 / 队列简单题
* [x] 配置并检查 PyTorch 环境
* [x] 验证 PyTorch 能够调用 NVIDIA GPU
* [x] 学习 Tensor 的创建、shape、dtype 和 device
* [x] 学习 Tensor 的索引、reshape 和广播机制
* [x] 理解 Tensor 与 NumPy array 的区别
* [x] 学习 autograd 自动求导
* [x] 理解 `requires_grad` 和计算图
* [x] 理解 `loss.backward()` 的作用
* [x] 学习梯度累积、梯度清零和参数更新
* [x] 使用 PyTorch 重写线性回归

### 今日学习记录

#### 1. PyTorch 环境配置

今天在 `ai-lab-prep` Conda 环境中安装并配置了 PyTorch。

当前环境：

* Python 版本：3.11.15
* PyTorch 版本：2.13.0+cu126
* PyTorch CUDA 版本：12.6
* GPU：NVIDIA GeForce RTX 4060 Laptop GPU

通过以下代码检查 CUDA 是否可用：

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

运行结果表明 PyTorch 能够正常识别并使用 GPU。

#### 2. Tensor 基础

学习了 Tensor 的创建方式，例如：

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
zeros = torch.zeros(2, 3)
ones = torch.ones(2, 3)
random_tensor = torch.randn(2, 3)
```

Tensor 的常见属性包括：

```python
print(x.shape)
print(x.ndim)
print(x.dtype)
print(x.device)
print(x.numel())
```

其中：

* `shape` 表示每个维度的大小
* `ndim` 表示 Tensor 的维数
* `dtype` 表示元素的数据类型
* `device` 表示 Tensor 位于 CPU 还是 GPU
* `numel()` 表示 Tensor 中元素的总数量

#### 3. reshape、索引与广播

使用 `reshape` 可以在元素总数不变的情况下改变 Tensor 的形状：

```python
x = torch.arange(12).reshape(3, 4)
```

使用索引和切片可以取出指定位置的数据：

```python
x[:, 1]
```

表示取出所有行的第二列。

广播允许不同 shape 的 Tensor 进行运算，例如：

```python
a = torch.ones(2, 3)
b = torch.tensor([10.0, 20.0, 30.0])

result = a + b
```

`b` 的 shape 为 `[3]`，可以广播为 `[2, 3]`，最终结果的 shape 为 `[2, 3]`。

#### 4. Tensor 与 NumPy array 的区别

Tensor 和 NumPy array 都支持：

* 多维数组
* shape
* 索引与切片
* 广播
* 矩阵运算

主要区别是：

* Tensor 可以在 GPU 上进行计算
* Tensor 支持 autograd 自动求导
* Tensor 是 PyTorch 模型参数、输入数据、预测结果和损失值的基础数据结构
* Tensor 具有 `device` 属性，可以在 CPU 和 GPU 之间移动

Tensor 移动到 GPU：

```python
x = x.to("cuda")
```

GPU Tensor 转为 NumPy array 时，需要先移动回 CPU：

```python
array = x.cpu().numpy()
```

如果 Tensor 属于计算图，则需要先使用 `detach()`：

```python
array = x.detach().cpu().numpy()
```

#### 5. autograd 自动求导

通过设置：

```python
x = torch.tensor(2.0, requires_grad=True)
```

可以让 PyTorch 追踪 `x` 参与的运算。

例如：

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2

y.backward()

print(x.grad)
```

因为：

$$
y=x^2
$$

所以：

$$
\frac{dy}{dx}=2x
$$

当 `x = 2` 时，最终梯度为 4。

`requires_grad=True` 表示 PyTorch 需要记录该 Tensor 参与的计算，以便后续计算最终结果对该 Tensor 的梯度。

#### 6. `loss.backward()` 的作用

`loss.backward()` 会从 loss 开始沿计算图反向传播，并通过链式法则计算 loss 对所有需要求导参数的梯度。

计算得到的梯度会存放在参数的 `.grad` 属性中：

```python
loss.backward()

print(weight.grad)
print(bias.grad)
```

`loss.backward()` 只负责计算梯度，不会自动更新模型参数。

参数需要根据梯度手动更新：

```python
with torch.no_grad():
    weight -= learning_rate * weight.grad
    bias -= learning_rate * bias.grad
```

#### 7. 梯度累积与清零

PyTorch 默认会累积梯度。

如果连续调用两次 `backward()`，第二次计算的梯度会加到第一次的梯度上。

因此每轮训练结束后需要清空梯度：

```python
weight.grad.zero_()
bias.grad.zero_()
```

使用优化器时，一般写成：

```python
optimizer.zero_grad()
```

#### 8. 使用 PyTorch 实现线性回归

今天使用 PyTorch 和 autograd 实现了线性回归，训练数据满足：

$$
y=3x+2+\text{noise}
$$

模型公式为：

$$
\hat{y}=xw+b
$$

代码中的前向计算：

```python
prediction = x @ weight + bias
```

损失函数使用均方误差：

```python
loss = ((prediction - y) ** 2).mean()
```

完整训练流程为：

```python
for epoch in range(epochs):
    prediction = x @ weight + bias
    loss = ((prediction - y) ** 2).mean()

    loss.backward()

    with torch.no_grad():
        weight -= learning_rate * weight.grad
        bias -= learning_rate * bias.grad

    weight.grad.zero_()
    bias.grad.zero_()
```

最终模型能够学习到接近真实值的参数：

```text
weight ≈ 3
bias ≈ 2
```


#### 9. 输入数据的 shape

在线性回归预测时使用：

```python
input_tensor = torch.tensor([[x]])
```

使用两层中括号是为了让输入保持二维结构。

当只有一个样本和一个特征时：

```text
shape = [1, 1]
```

其中：

* 第一维表示样本数量
* 第二维表示每个样本的特征数量

训练数据通常使用：

```text
[batch_size, feature_count]
```

的格式。

### 遇到的问题

* 需要注意 CPU Tensor 和 GPU Tensor 不能直接进行运算
* GPU Tensor 不能直接转换为 NumPy array，需要先调用 `.cpu()`
* 参与自动求导的 Tensor 转换为 NumPy 前，需要先调用 `.detach()`
* PyTorch 默认累积梯度，每轮参数更新后必须清空梯度
* 理解了预测单个数据时使用 `[[x]]` 是为了保持 `[样本数, 特征数]` 的二维格式

### 今日总结

今天完成了 PyTorch 入门的第一部分，掌握了 Tensor 的基本操作、shape、broadcast、device 和 autograd。

目前能够解释：

* Tensor 和 NumPy array 的区别
* `requires_grad=True` 的含义
* `loss.backward()` 做了什么
* 为什么训练时需要清空梯度
* PyTorch 线性回归的基本训练流程

### 明日计划

*  完成一道二分查找简单题
*  学习 `Dataset` 的作用和基本使用方法
*  学习 `DataLoader` 的作用和基本使用方法
*  理解 batch、batch size 和数据打乱
*  使用 `torchvision` 加载 MNIST 或 Fashion-MNIST
*  理解图像数据 `[batch_size, channels, height, width]` 的维度含义
*  使用 Matplotlib 可视化 16 张图片



