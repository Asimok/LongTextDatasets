"""
统计
    长答案长度
    文章长度
    长答案在文章中的占比
    长答案在文章中是不是完整句子
"""
import json

from tqdm import tqdm

dataset_path = '/data2/maqi/datasets/NQ/rawNQ/v1.0-simplified-nq-train.jsonl'
with open(dataset_path, 'r') as f:
    data_set = f.readlines()
multi_span = 0
for temp_data in tqdm(data_set):
    json_data = json.loads(temp_data)
    annotations = json_data['annotations']
    for annotation in annotations:
        if len(annotation['short_answers']) > 1:
            multi_span += 1
print(multi_span)
