import copy
import json

import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm

from tag_BIO_for_NQ import process_dataset


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


def clean_datasets(dataset_path, _output_path):
    with open(dataset_path, 'r') as f:
        data_set = f.readlines()
    clean_data_set = []
    for temp_data in tqdm(data_set):
        json_data = json.loads(temp_data)
        backup_json_data = copy.deepcopy(json_data)
        # 正则匹配 清除html标签
        document_text = json_data['document_text']
        original_tokens = document_text.split()
        clean_html_text = BeautifulSoup(document_text, 'html.parser').get_text()
        tokens = clean_html_text.split()
        clean_html_text = ' '.join(tokens)
        # 还原完整答案
        # long_answer_candidates = json_data['long_answer_candidates']
        annotations = json_data['annotations']
        new_annotations = []
        for annotation in annotations:
            backup_annotation = copy.deepcopy(annotation)
            if annotation['long_answer']['start_token'] != -1:
                long_ans_str_tokens = BeautifulSoup(' '.join(original_tokens[
                                                             annotation['long_answer']['start_token']:
                                                             annotation['long_answer']['end_token']]),
                                                    'html.parser').get_text().split()
                new_index = get_new_index(clean_tokens=tokens, clean_ans_tokens=long_ans_str_tokens)
                if new_index is not None:
                    backup_annotation['long_answer']['start_token'] = new_index[0]
                    backup_annotation['long_answer']['end_token'] = new_index[1]
                else:
                    continue
            short_answers = []
            for short_answer in annotation['short_answers']:
                backup_short_answer = copy.deepcopy(short_answer)
                if short_answer['start_token'] != -1:
                    short_ans_str_tokens = BeautifulSoup(' '.join(original_tokens[
                                                                  short_answer['start_token']:
                                                                  short_answer['end_token']]),
                                                         'html.parser').get_text().split()
                    new_index = get_new_index(clean_tokens=tokens, clean_ans_tokens=short_ans_str_tokens)
                    if new_index is not None:
                        backup_short_answer['start_token'] = new_index[0]
                        backup_short_answer['end_token'] = new_index[1]
                    else:
                        continue
                short_answers.append(backup_short_answer)
                backup_annotation['short_answers'] = short_answers

            new_annotations.append(backup_annotation)
        backup_json_data['annotations'] = new_annotations
        backup_json_data['document_text'] = clean_html_text
        # 去掉无用字段
        backup_json_data.pop('long_answer_candidates')
        backup_json_data.pop('document_url')
        clean_data_set.append(backup_json_data)
    with open(_output_path, 'w') as f:
        for line in clean_data_set:
            f.write(json.dumps(line, cls=NpEncoder))
            f.write('\n')
    print('清洗结束')


if __name__ == '__main__':

    # exp
    # dev_dataset_path = '/data2/maqi/datasets/NQ/less_v1.0-simplified-nq-train.jsonl'
    # dev_clean_output_path = '/data2/maqi/datasets/NQ/test-clean-nq-dev.jsonl'
    # dev_tag_output_path ='/data2/maqi/datasets/NQ/test-tag-nq-dev.jsonl'

    # dev
    # dev_dataset_path = '/data2/maqi/datasets/NQ/rawNQ/v1.0-simplified-nq-dev.jsonl'
    # dev_clean_output_path = '/data2/maqi/datasets/NQ/cleanNQ/clean-nq-dev.jsonl'
    # dev_tag_output_path ='/data2/maqi/datasets/NQ/tagNQ/tag-nq-dev.jsonl'
    #
    # test
    test_dataset_path = '/data2/maqi/datasets/NQ/rawNQ/v1.0-simplified-nq-test.jsonl'
    test_clean_output_path = '/data2/maqi/datasets/NQ/cleanNQ/clean-nq-test.jsonl'
    test_tag_output_path ='/data2/maqi/datasets/NQ/tagNQ/tag-nq-test.jsonl'

    clean_datasets(dataset_path=test_dataset_path, _output_path=test_clean_output_path)
    process_dataset(_read_dataset_path=test_clean_output_path, _output_path=test_tag_output_path, labels_scheme="IO")