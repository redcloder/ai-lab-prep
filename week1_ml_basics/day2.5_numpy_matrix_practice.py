import numpy as np


def section_1_shape_and_reshape():
    print("===== 1. shape and reshape =====")

    a = np.array([1, 2, 3])
    print("a:")
    print(a)
    print("a.shape:", a.shape) #(3,)

    a_col = a.reshape(3, 1)
    print("\na_col:")
    print(a_col) #[[1],[2],[3]]
    print("a_col.shape:", a_col.shape)

    a_row = a.reshape(1, 3)
    print("\na_row:")
    print(a_row)
    print("a_row.shape:", a_row.shape)


def section_2_transpose():
    print("\n===== 2. transpose =====")

    A = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    print("A:")
    print(A)
    print("A.shape:", A.shape)

    print("\nA.T:")
    print(A.T)
    print("A.T.shape:", A.T.shape)

    a = np.array([1, 2, 3])
    print("\n1D array a.shape:", a.shape)
    print("1D array a.T.shape:", a.T.shape)

    a_col = a.reshape(3, 1)
    print("\na_col.shape:", a_col.shape)
    print("a_col.T.shape:", a_col.T.shape)


def section_3_matrix_multiplication():
    print("\n===== 3. matrix multiplication =====")

    A = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    B = np.array([
        [10, 20],
        [30, 40],
        [50, 60]
    ])

    C = A @ B

    print("A.shape:", A.shape)
    print("B.shape:", B.shape)
    print("C = A @ B:")
    print(C)
    print("C.shape:", C.shape)


def section_4_star_vs_at():
    print("\n===== 4. * vs @ =====")

    A = np.array([
        [1, 2],
        [3, 4]
    ])

    B = np.array([
        [10, 20],
        [30, 40]
    ])

    print("A * B:")
    print(A * B)

    print("\nA @ B:")
    print(A @ B)


def section_5_ml_example():
    print("\n===== 5. machine learning example: X @ w + b =====")

    # 4 个样本，每个样本 3 个特征
    X = np.array([
        [2, 10, 7],
        [3, 15, 6],
        [5, 20, 8],
        [1, 5, 5]
    ])

    # 3 个特征，对应 3 个权重
    w = np.array([
        [1.5],
        [0.8],
        [2.0]
    ])

    b = 1.0

    z = X @ w + b

    print("X.shape:", X.shape)
    print("w.shape:", w.shape)
    print("z = X @ w + b:")
    print(z)
    print("z.shape:", z.shape)


def main():
    section_1_shape_and_reshape()
    section_2_transpose()
    section_3_matrix_multiplication()
    section_4_star_vs_at()
    section_5_ml_example()


if __name__ == "__main__":
    main()