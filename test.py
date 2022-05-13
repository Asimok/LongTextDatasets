class Solution:
    def largestGoodInteger(self, num: str) -> str:
        ans = []
        if len(num) == 3:
            if num[0] == num[1] and num[1] == num[2]:
                return num

        for i in range(len(num) - 2):
            if num[i] == num[i + 1] and num[i + 1] == num[i + 2]:
                ans.append(num[i:i + 3])
        if len(ans) == 0:
            return ""
        max_val = int(ans[0])
        for j in ans:
            temp = int(j)
            if temp > max_val:
                max_val = temp
        if max_val == 0:
            return "000"
        return str(max_val)


a ="6777133339"
c = Solution()
d = c.largestGoodInteger(a)
print(d)
