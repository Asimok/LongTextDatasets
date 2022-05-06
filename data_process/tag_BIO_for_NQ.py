"""
预处理：
    去除问题中的停止词
    转为小写标注
学习目标：
    输入： 长文本 问题
    输出： 短文本
标记策略：
    train/dev: IOSL
    test: IO
标注优先级：I S L (短答案中的token一定在长答案中出现)
-I 问题中出现的token
-S 短答案中的token
-L 长答案中的token
-O 其他不相关词
"""
import copy
import json
from string import punctuation

import nltk
import numpy as np
from nltk.corpus import stopwords
from tqdm import tqdm

stopwords = stopwords.words('english')
for w in punctuation:
    stopwords.append(w)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def NLTK_utils(word_str, _stopwords):
    word_list = word_str.split()
    # 去重
    word_list = list(set(word_list))
    filtered_words = [word for word in word_list if word not in _stopwords]
    return filtered_words


def _create_sequence_labels(labels_scheme, spans_IO, spans_S, spans_L, n_labels):
    """
    :param labels_scheme: "IOSL"  "IO"
    :param spans_IO:
    :param spans_S:
    :param spans_L:
    :param n_labels:
    :return:
    """
    labels_scheme = labels_scheme
    labels = {
        'O': 0,
        'I': 1,
        'S': 2,
        'L': 3
    }
    # initialize all labels to O
    labeling = [labels['O']] * n_labels

    def IO_labels(spans):
        for start, end in spans:
            # create I labels
            labeling[start: end + 1] = [labels['I']] * (end - start + 1)
        return labeling

    def SL_labels(spans, label):
        # 在IO的基础上标注S/L S优先级高于L
        for start, end in spans:
            # create S/L labels
            for i in range(start, end + 1):
                if labeling[i] == labels['O']:
                    labeling[i] = label
        return labeling

    if labels_scheme == "IOSL":
        if spans_IO is not None:
            IO_labels(spans=spans_IO)
        if spans_S is not None:
            SL_labels(spans=spans_S, label=labels['S'])
        if spans_L is not None:
            SL_labels(spans=spans_L, label=labels['L'])
    elif labels_scheme == "IO":
        if spans_IO is not None:
            IO_labels(spans=spans_IO)
    else:
        raise Exception("Illegal labeling scheme")
    return labeling


def generate_question_spans(question, passage):
    """
    找到 question 中 token 在 passage 中出现的所有位置
    :param question:
    :param passage:
    :return:
    """
    if len(question) == 0:
        return None
    questions_spans = []
    passage_token_np = np.array(passage)
    for temp_question_token in question:
        # 查找所有出现的位置
        loc = np.where(passage_token_np == temp_question_token)
        for idx in list(loc[0]):
            questions_spans.append((idx, idx))
    questions_spans = list(set(questions_spans))
    return questions_spans


def process_dataset(_read_dataset_path, _output_path, labels_scheme):
    with open(_read_dataset_path, 'r') as f:
        data_set = f.readlines()
    clean_data_set = []
    for temp_data in tqdm(data_set):
        json_data = json.loads(temp_data)
        backup_json_data = copy.deepcopy(json_data)
        document_text = json_data['document_text'].lower()
        question_text = json_data['question_text'].lower()
        annotations = json_data['annotations']
        original_tokens = document_text.split()
        # question_text  去停用词
        question_tokens = NLTK_utils(word_str=question_text, _stopwords=stopwords)

        long_ans_spans = []
        short_ans_spans = []
        for annotation in annotations:
            if annotation['long_answer']['start_token'] != -1:
                long_ans_spans.append(
                    (annotation['long_answer']['start_token'], annotation['long_answer']['end_token']))
            for short_answer in annotation['short_answers']:
                if short_answer['start_token'] != -1:
                    short_ans_spans.append((short_answer['start_token'], short_answer['end_token']))
        # generate_question_spans
        question_spans = generate_question_spans(question=question_tokens, passage=original_tokens)
        backup_json_data['tag'] = _create_sequence_labels(labels_scheme=labels_scheme, spans_IO=question_spans,
                                                          spans_S=short_ans_spans, spans_L=long_ans_spans,
                                                          n_labels=len(original_tokens))
        # 去掉无用字段
        clean_data_set.append(backup_json_data)
    with open(_output_path, 'w') as f:
        for line in clean_data_set:
            f.write(json.dumps(line, cls=NpEncoder))
            f.write('\n')
    print('标注结束！')


if __name__ == '__main__':
    read_dataset_path = '/data2/maqi/datasets/NQ/clean-nq-train.jsonl'
    output_path = '/data2/maqi/datasets/NQ/tag-clean-nq-train.jsonl'
    process_dataset(_read_dataset_path=read_dataset_path, _output_path=output_path, labels_scheme="IOSL")
