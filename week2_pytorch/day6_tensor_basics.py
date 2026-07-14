import numpy as np
import torch


def main() -> None:
    # 1. 直接从 Python 列表创建 Tensor
    scalar = torch.tensor(5.0)
    vector = torch.tensor([1.0, 2.0, 3.0])
    matrix = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    print("=== Tensor 的维度 ===")
    print("scalar:", scalar)
    print("scalar shape:", scalar.shape)
    print("scalar ndim:", scalar.ndim)

    print("\nvector:", vector)
    print("vector shape:", vector.shape)
    print("vector ndim:", vector.ndim)

    print("\nmatrix:")
    print(matrix)
    print("matrix shape:", matrix.shape)
    print("matrix ndim:", matrix.ndim)

    # 2. 常见的 Tensor 创建函数
    zeros = torch.zeros(2, 3)
    ones = torch.ones(2, 3)
    random_tensor = torch.rand(2, 3)
    normal_tensor = torch.randn(2, 3)
    integer_range = torch.arange(0, 10, 2)

    print("\n=== 常见创建方式 ===")
    print("zeros:\n", zeros)
    print("ones:\n", ones)
    print("rand:\n", random_tensor)
    print("randn:\n", normal_tensor)
    print("arange:", integer_range)

    # 3. dtype：Tensor 中元素的数据类型
    float_tensor = torch.tensor([1.0, 2.0, 3.0])
    int_tensor = torch.tensor([1, 2, 3])
    explicit_float64 = torch.tensor(
        [1.0, 2.0, 3.0],
        dtype=torch.float64,
    )

    print("\n=== 数据类型 ===")
    print("float_tensor dtype:", float_tensor.dtype)
    print("int_tensor dtype:", int_tensor.dtype)
    print("explicit_float64 dtype:", explicit_float64.dtype)

    # 类型转换
    converted = int_tensor.float()
    print("converted dtype:", converted.dtype)

    # 4. 索引和切片
    x = torch.tensor([
        [10, 20, 30],
        [40, 50, 60],
    ])

    print("\n=== 索引和切片 ===")
    print("x:\n", x)
    print("第一行:", x[0])
    print("第二行第三列:", x[1, 2])
    print("第二列:", x[:, 1])

    # 5. reshape
    y = torch.arange(12)

    print("\n=== reshape ===")
    print("原始 y:", y)
    print("原始 shape:", y.shape)

    y_3x4 = y.reshape(3, 4)
    print("reshape 为 3×4:\n", y_3x4)
    print("新 shape:", y_3x4.shape)

    # -1 表示让 PyTorch 自动计算这一维
    y_auto = y.reshape(2, -1)
    print("reshape 为 2×自动计算:\n", y_auto)
    print("新 shape:", y_auto.shape)

    # 6. NumPy 和 Tensor 相互转换
    numpy_array = np.array([1.0, 2.0, 3.0])
    tensor_from_numpy = torch.from_numpy(numpy_array)
    numpy_from_tensor = tensor_from_numpy.numpy()

    print("\n=== NumPy 相互转换 ===")
    print("NumPy array:", numpy_array)
    print("Tensor:", tensor_from_numpy)
    print("转换回 NumPy:", numpy_from_tensor)

    matrix = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    #7.广播机制
    row = torch.tensor([10.0, 20.0, 30.0])

    result = matrix + row

    print("\n=== broadcasting ===")
    print("matrix shape:", matrix.shape)
    print("row shape:", row.shape)
    print("result:")
    print(result)


if __name__ == "__main__":
    main()