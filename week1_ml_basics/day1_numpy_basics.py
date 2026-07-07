import numpy as np


def main():
    # 1. 创建数组
    a = np.array([1, 2, 3, 4])
    print("a =", a)
    print("a.shape =", a.shape)

    # 2. 创建二维矩阵
    b = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    print("b =",end=" ")
    print(b)
    print("b.shape =", b.shape)

    # 3. reshape 改变形状
    c = np.arange(12)
    print("c =", c)

    c_reshaped = c.reshape(3, 4)
    print("c_reshaped =")
    print(c_reshaped)
    print("c_reshaped.shape =", c_reshaped.shape)

    # 4. 矩阵基本运算
    x = np.array([1, 2, 3])
    y = np.array([10, 20, 30])

    print("x + y =", x + y)
    print("x * y =", x * y)

    # 5. 广播机制
    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    bias = np.array([10, 20, 30])

    result = matrix + bias

    print("matrix + bias =")
    print(result)

    x = np.array([[10], [20]])
    print("matrix + x =")
    print(matrix + x)

    # 6. 常用统计操作
    print("matrix.sum() =", matrix.sum())
    print("matrix.mean() =", matrix.mean())
    print("matrix.sum(axis=0) =", matrix.sum(axis=0))#按列
    print("matrix.sum(axis=1) =", matrix.sum(axis=1))#按行


if __name__ == "__main__":
    main()