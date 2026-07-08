from typing import List
import ast


def contains(nums: List[int]) -> bool:
    seen = {}

    for i, num in enumerate(nums):
        if num in seen:
            return True

        seen[num] = i

    return False


def main():
    nums_input = input("请输入数组，例如：[2, 7, 11, 15]：")
    nums = ast.literal_eval(nums_input)
    print(contains(nums))



if __name__ == "__main__":
    main()