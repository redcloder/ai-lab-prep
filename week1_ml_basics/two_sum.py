from typing import List


def two_sum(nums: List[int], target: int) -> List[int]:
    seen = {}

    for i, num in enumerate(nums):
        need = target - num

        if need in seen:
            return [seen[need], i]

        seen[num] = i

    return []


def main():
    nums_input = input("请输入数组，用空格分隔，例如：2 7 11 15：")
    target_input = input("请输入 target，例如：9：")

    nums = list(map(int, nums_input.split()))
    target = int(target_input)

    result = two_sum(nums, target)
    print("结果是：", result)


if __name__ == "__main__":
    main()