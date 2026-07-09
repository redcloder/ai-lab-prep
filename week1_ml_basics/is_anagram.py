class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False

        count = {}

        for ch in s:
            count[ch] = count.get(ch, 0) + 1

        for ch in t:
            if ch not in count:
                return False
            count[ch] -= 1
            if count[ch] < 0:
                return False

        return True


def main():
    solution = Solution()

    test_cases = [
        ("anagram", "nagaram", True),
        ("rat", "car", False),
        ("a", "a", True),
        ("a", "b", False),
        ("ab", "ba", True),
        ("abc", "ab", False),
        ("aa", "a", False),
        ("aacc", "ccac", False),
    ]

    for s, t, expected in test_cases:
        result = solution.isAnagram(s, t)
        print(f"s = {s}, t = {t}, result = {result}, expected = {expected}")

        if result == expected:
            print("测试通过")
        else:
            print("测试失败")

        print("-" * 40)


if __name__ == "__main__":
    main()