def two_sum_sorted(numbers, target):
    """
    在升序数组 numbers 中寻找两个数，使它们的和等于 target。

    返回值：
        找到时：返回两个下标，使用 LeetCode 167 的 1-based 下标
        找不到：返回 None

    时间复杂度：O(n)
    空间复杂度：O(1)
    """
    left = 0
    right = len(numbers) - 1

    while left < right:
        current_sum = numbers[left] + numbers[right]


        if current_sum == target:
            return [left + 1, right + 1]

        if current_sum < target:
            # 当前和太小，需要让和变大
            left += 1
        else:
            # 当前和太大，需要让和变小
            right -= 1

    return None


def read_numbers():
    """
    从终端读取一个升序整数数组。

    输入示例：
        2 7 11 15
    """
    while True:
        raw = input(
            "请输入升序整数数组（空格分隔，输入 q 退出）："
        ).strip()

        if raw.lower() == "q":
            return None

        try:
            numbers = [int(x) for x in raw.split()]
        except ValueError:
            print("输入错误：数组中只能包含整数。\n")
            continue

        if len(numbers) < 2:
            print("输入错误：数组至少需要两个数字。\n")
            continue

        if numbers != sorted(numbers):
            print("输入错误：数组必须按从小到大排列。\n")
            continue

        return numbers


def read_target():
    """从终端读取目标值。"""
    while True:
        raw = input("请输入 target：").strip()

        if raw.lower() == "q":
            return None

        try:
            return int(raw)
        except ValueError:
            print("输入错误：target 必须是整数。")


def main():
    print("=" * 50)
    print("有序数组两数之和：双指针测试程序")
    print("数组和 target 输入位置均可输入 q 退出")
    print("=" * 50)

    while True:
        numbers = read_numbers()

        if numbers is None:
            break

        target = read_target()

        if target is None:
            break

        result = two_sum_sorted(numbers, target)

        if result is None:
            print("\n结果：没有找到满足条件的两个数。")
        else:
            index1, index2 = result

            # 返回值是 1-based 下标
            # 访问 Python 数组时需要减 1
            value1 = numbers[index1 - 1]
            value2 = numbers[index2 - 1]

            print("\n找到答案：")
            print(f"1-based 下标：{result}")
            print(
                f"对应数字：{value1} + {value2} = {target}"
            )

        print("\n" + "-" * 50 + "\n")

    print("程序结束。")


if __name__ == "__main__":
    main()