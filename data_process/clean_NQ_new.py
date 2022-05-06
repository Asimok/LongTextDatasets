import copy
import json
from string import punctuation
from sys import argv

import nltk
import nltk.data
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from tqdm import tqdm

stopwords = stopwords.words('english')
for w in punctuation:
    stopwords.append(w)


def splitSentence(paragraph):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(paragraph)
    return sentences


def NLTK_utils(word_str, _stopwords):
    word_list = word_str.split()
    # 去重
    word_list = list(set(word_list))
    filtered_words = [word for word in word_list if word not in _stopwords]
    return filtered_words


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


def spanSentence(split_document_text):
    """
    返回切割后句子在 原文 和 清洗后文档 中token的起止位置
    :param split_document_text: 分割的原文句子
    :return:
        1.清洗后句子 以及 在 原文和 清洗后文档 中的位置
        2.清洗后句子结尾的token下标
    """
    org_start = 0
    org_end = 0
    cur_start = 0
    cur_end = 0
    ans = []
    sen_end_idx = []
    clean_sens = []
    for sen in split_document_text:
        sen_token = sen.split()
        org_end = org_end + len(sen_token)

        clean_sen = BeautifulSoup(sen, 'html.parser').get_text()
        clean_sen_tokens = clean_sen.split()
        clean_sen = ' '.join(clean_sen_tokens)
        clean_sens.append(clean_sen)

        cur_end = cur_end + len(clean_sen_tokens)
        ans.append({'sentence': clean_sen, 'org_start_token': org_start, 'org_end_token': org_end,
                    'cur_start_token': cur_start, 'cur_end_token': cur_end})
        sen_end_idx.append(cur_end)
        org_start = org_end
        cur_start = cur_end
    return ans, sen_end_idx, clean_sens


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


def get_new_index(clean_tokens, clean_ans_tokens):
    if len(clean_ans_tokens) == 0:
        return None
    # 先找到首元素的索引
    clean_tokens_np = np.array(clean_tokens)
    result = np.where(clean_tokens_np == clean_ans_tokens[0])
    len_ans = len(clean_ans_tokens)
    clean_ans_str = ' '.join(clean_ans_tokens)
    # 探测后续位置是否匹配
    for first_token in list(result[0]):
        find_ans = ' '.join(clean_tokens[first_token:first_token + len_ans])
        if find_ans == clean_ans_str:
            return [first_token, first_token + len_ans]
    return None


# BinarySearch：find the location of the target number
def BinarySearch(nums: list, x: int):
    """
        nums: Sorted array from smallest to largest
        x: Target number
        查找nums中第一个大于x的数字的下标
    """
    left, right = 0, len(nums) - 1
    res = right
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] > x:
            res = mid
            right = mid - 1
        else:
            left = mid + 1
    if x <= nums[res]:
        return res
    return None


def find_sentence(new_index, sentence_end_idx):
    start = new_index[0]
    end = new_index[1]
    ans = []
    first_sen_idx = BinarySearch(sentence_end_idx, start)
    end_sen_idx = BinarySearch(sentence_end_idx, end)
    for i in range(first_sen_idx, end_sen_idx + 1):
        ans.append(i)
    return ans


def clean_datasets(dataset_path, _output_path, labels_scheme):
    MORE_SHORT = 0

    with open(dataset_path, 'r') as f:
        data_set = f.readlines()
    clean_data_set = []
    for temp_data in tqdm(data_set):
        json_data = json.loads(temp_data)
        backup_json_data = copy.deepcopy(json_data)
        # 去掉无用字段
        backup_json_data.pop('long_answer_candidates')
        backup_json_data.pop('document_url')
        # 正则匹配 清除html标签
        document_text = json_data['document_text']
        original_tokens = document_text.split()
        clean_html_text = BeautifulSoup(document_text, 'html.parser').get_text()
        # 清除空格
        current_tokens = clean_html_text.split()
        clean_html_text = ' '.join(current_tokens)
        # 拆分成句子
        split_document_text = splitSentence(document_text)
        # 标注每个原始句子在原文中的起始位置
        split_document_text_spans, sentence_end_idx, clean_split_document_text_spans = spanSentence(split_document_text)

        # 还原完整答案
        # 保留答案原始位置 增加所在句子标注

        annotations = json_data['annotations']
        new_annotations = []
        long_ans_spans = []
        short_ans_spans = []
        for annotation in annotations:
            backup_annotation = copy.deepcopy(annotation)

            if annotation['long_answer']['start_token'] != -1:
                long_ans_str_tokens = BeautifulSoup(' '.join(original_tokens[
                                                             annotation['long_answer']['start_token']:
                                                             annotation['long_answer']['end_token']]),
                                                    'html.parser').get_text().split()
                new_index = get_new_index(clean_tokens=current_tokens, clean_ans_tokens=long_ans_str_tokens)
                if new_index is not None:
                    # 增加所在句子标签
                    long_ans_sentence_idx = find_sentence(new_index, sentence_end_idx)

                    backup_annotation['long_answer']['cur_start_token'] = new_index[0]
                    backup_annotation['long_answer']['cur_end_token'] = new_index[1]
                    backup_annotation['long_answer']['org_start_token'] = backup_annotation['long_answer'][
                        'start_token']
                    backup_annotation['long_answer']['org_end_token'] = backup_annotation['long_answer']['end_token']
                    backup_annotation['long_answer']['sentence_idx'] = long_ans_sentence_idx
                    # 删除原来答案标注键值对
                    backup_annotation['long_answer'].pop('start_token')
                    backup_annotation['long_answer'].pop('end_token')
                    # 记录span位置
                    long_ans_spans.append((new_index[0], new_index[1]))
            else:
                backup_annotation['long_answer']['cur_start_token'] = -1
                backup_annotation['long_answer']['cur_end_token'] = -1
                backup_annotation['long_answer']['org_start_token'] = -1
                backup_annotation['long_answer']['org_end_token'] = -1
                backup_annotation['long_answer']['sentence_idx'] = []
                # 删除原来答案标注键值对
                backup_annotation['long_answer'].pop('start_token')
                backup_annotation['long_answer'].pop('end_token')

            short_answers = []
            for short_answer in annotation['short_answers']:
                backup_short_answer = copy.deepcopy(short_answer)

                if short_answer['start_token'] != -1:
                    short_ans_str_tokens = BeautifulSoup(' '.join(original_tokens[
                                                                  short_answer['start_token']:
                                                                  short_answer['end_token']]),
                                                         'html.parser').get_text().split()
                    new_index = get_new_index(clean_tokens=current_tokens, clean_ans_tokens=short_ans_str_tokens)
                    if new_index is not None:
                        short_ans_sentence_idx = find_sentence(new_index, sentence_end_idx)
                        # if len(short_ans_sentence_idx) != 1:
                        #     print(' '.join(short_ans_str_tokens))

                        backup_short_answer['cur_start_token'] = new_index[0]
                        backup_short_answer['cur_end_token'] = new_index[1]
                        backup_short_answer['org_start_token'] = new_index[0]
                        backup_short_answer['org_end_token'] = new_index[1]
                        backup_short_answer['sentence_idx'] = short_ans_sentence_idx[0]
                        # 删除原来答案标注键值对
                        backup_short_answer.pop('start_token')
                        backup_short_answer.pop('end_token')
                        short_ans_spans.append((new_index[0], new_index[1]))
                else:
                    backup_short_answer['cur_start_token'] = -1
                    backup_short_answer['cur_end_token'] = -1
                    backup_short_answer['org_start_token'] = -1
                    backup_short_answer['org_end_token'] = -1
                    backup_short_answer['sentence_idx'] = []
                    # 删除原来答案标注键值对
                    backup_short_answer.pop('start_token')
                    backup_short_answer.pop('end_token')

                short_answers.append(backup_short_answer)
                backup_annotation['short_answers'] = short_answers

            new_annotations.append(backup_annotation)
        # TODO 根据 new_annotations questions 打标签

        lower_document_text = clean_html_text.lower()
        question_text = json_data['question_text'].lower()
        lower_original_tokens = lower_document_text.split()
        # question_text  去停用词
        question_tokens = NLTK_utils(word_str=question_text, _stopwords=stopwords)

        # generate_question_spans
        question_spans = generate_question_spans(question=question_tokens, passage=lower_original_tokens)

        tag_list = _create_sequence_labels(labels_scheme=labels_scheme, spans_IO=question_spans,
                                           spans_S=short_ans_spans, spans_L=long_ans_spans,
                                           n_labels=len(lower_original_tokens))
        # 标签分配给每个句子
        tag_start = 0
        for sen_idx in range(len(split_document_text_spans)):
            tag_end = sentence_end_idx[sen_idx]
            split_document_text_spans[sen_idx]['tag'] = tag_list[tag_start:tag_end]
            tag_start = tag_end

        backup_json_data['annotations'] = new_annotations
        backup_json_data['document_text'] = clean_html_text
        backup_json_data['sentences'] = split_document_text_spans

        clean_data_set.append(backup_json_data)
    print('开始存储...')
    with open(_output_path, 'w') as f:
        for line in clean_data_set:
            f.write(json.dumps(line, cls=NpEncoder))
            f.write('\n')
    print('数据集处理结束')


if __name__ == '__main__':

    mode = argv[1]
    # mode = "dev"
    print('模式：', mode)

    # exp
    # test_dataset_path = '/data2/maqi/datasets/NQ/less_v1.0-simplified-nq-train.jsonl'
    # test_clean_output_path = '/data2/maqi/datasets/NQ/test-clean-nq-dev.jsonl'
    # test_tag_output_path = '/data2/maqi/datasets/NQ/test-tag-nq-dev.jsonl'

    # train
    # train 太大 需要切片
    train_dataset_path_chunk1 = '/data2/maqi/datasets/NQ/rawNQ/train_chunks/chunk1_v1.0-simplified-nq-train.jsonl'
    train_dataset_path_chunk2 = '/data2/maqi/datasets/NQ/rawNQ/train_chunks/chunk2_v1.0-simplified-nq-train.jsonl'
    train_dataset_path_chunk3 = '/data2/maqi/datasets/NQ/rawNQ/train_chunks/chunk3_v1.0-simplified-nq-train.jsonl'
    train_dataset_path_chunk4 = '/data2/maqi/datasets/NQ/rawNQ/train_chunks/chunk4_v1.0-simplified-nq-train.jsonl'

    train_output_path_chunk1 = '/data2/maqi/datasets/NQ/cleanAndTagNQ/chunk1-clean-tag-nq-train.jsonl'
    train_output_path_chunk2 = '/data2/maqi/datasets/NQ/cleanAndTagNQ/chunk2-clean-tag-nq-train.jsonl'
    train_output_path_chunk3 = '/data2/maqi/datasets/NQ/cleanAndTagNQ/chunk3-clean-tag-nq-train.jsonl'
    train_output_path_chunk4 = '/data2/maqi/datasets/NQ/cleanAndTagNQ/chunk4-clean-tag-nq-train.jsonl'

    # dev
    dev_dataset_path = '/data2/maqi/datasets/NQ/rawNQ/v1.0-simplified-nq-dev.jsonl'
    dev_output_path = '/data2/maqi/datasets/NQ/cleanAndTagNQ/clean-tag-nq-dev.jsonl'

    # test
    test_dataset_path = '/data2/maqi/datasets/NQ/rawNQ/v1.0-simplified-nq-test.jsonl'
    test_output_path = '/data2/maqi/datasets/NQ/cleanAndTagNQ/clean-tag-nq-test.jsonl'

    if mode == "train_chunk1":
        clean_datasets(dataset_path=train_dataset_path_chunk1, _output_path=train_output_path_chunk1, labels_scheme="IOSL")
    elif mode == "train_chunk2":
        clean_datasets(dataset_path=train_dataset_path_chunk2, _output_path=train_output_path_chunk2,
                       labels_scheme="IOSL")
    elif mode == "train_chunk3":
        clean_datasets(dataset_path=train_dataset_path_chunk3, _output_path=train_output_path_chunk3,
                       labels_scheme="IOSL")
    elif mode == "train_chunk4":
        clean_datasets(dataset_path=train_dataset_path_chunk4, _output_path=train_output_path_chunk4,
                       labels_scheme="IOSL")
    elif mode == "dev":
        clean_datasets(dataset_path=dev_dataset_path, _output_path=dev_output_path, labels_scheme="IOSL")
    elif mode == "test":
        clean_datasets(dataset_path=test_dataset_path, _output_path=test_output_path, labels_scheme="IO")
