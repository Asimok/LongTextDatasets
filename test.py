class Solution:

    def appealSum(self, s: str) -> int:

        def create_ngram_list(input_list, ngram_num):
            ngram_list = []
            if len(input_list) <= ngram_num:
                ngram_list.append(input_list)
            else:
                for tmp in zip(*[input_list[i:] for i in range(ngram_num)]):
                    tmp = "".join(tmp)
                    ngram_list.append(tmp)
            sum = 0
            for l in ngram_list:
                sum += len(set(list(l)))
            return sum

        ans = 0
        for i in range(len(s)):
            ans += create_ngram_list(s, i+1)
        return ans


a = "abbca"
c = Solution()
d = c.appealSum(a)
print(d)
