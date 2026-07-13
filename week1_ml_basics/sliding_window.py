def find_max_average(nums: list[int], k: int) -> float:
    if k > len(nums):
        return 0
    if k <=0:
        return 0
    window_sum = sum(nums[:k])
    max_sum = window_sum

    for right in range(k, len(nums)):
        window_sum += nums[right]
        window_sum -= nums[right - k]
        max_sum = max(max_sum, window_sum)

    return max_sum / k

def main():
    raw = input("请输入整数数组,空格分隔").strip()
    try:
        numbers = [int(x) for x in raw.split()]
    except ValueError:
        print("输入错误：数组中只能包含整数。\n")

    target = int(input("请输入滑动窗口的大小").strip())

    max_average = find_max_average(numbers, target)
    if max_average == 0:
        print("target输入错误")
        return
    print(max_average)

if __name__ == "__main__":
    main()
