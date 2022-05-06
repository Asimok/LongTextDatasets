import json

from tqdm import tqdm

read_path = '/data2/wangbingchao/dataset/TriviaQA-rc/squad/wikipedia-train.json'
# read_path='/data2/wangbingchao/dataset/TriviaQA-rc/squad/wikipedia-dev.json'
# with open(read_path,'r') as f:
#     data = f.readlines()

json_data = json.load(open(read_path, 'r'))
data = json_data['data']
total = 0
http_ans = 0
for temp_data in tqdm(data):
    for paragraphs in temp_data['paragraphs']:
        for qas in paragraphs['qas']:
            total+=1
            if 'www' in  qas['question']:
                http_ans+=1

