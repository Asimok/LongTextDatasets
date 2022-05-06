def create_ngram_list(input_list, ngram_num):
    ngram_list = []
    if len(input_list) <= ngram_num:
        ngram_list.append(input_list)
    else:
        for tmp in zip(*[input_list[i:] for i in range(ngram_num)]):
            tmp = "".join(tmp)
            ngram_list.append(tmp)
    return ngram_list
s = "abbca"
b = create_ngram_list(s,3)