import collections
import json
import os.path
from tqdm import tqdm

dataset_dir = '/data2/wangbingchao/dataset/HotpotQA'
train_file = 'hotpot_train_v1.1.json'
dev_file = 'hotpot_dev_distractor_v1.json'


def hotpotToSquad(hotpot: str, squad: str):
    count = {"YesNo": 0, "normal": 0}
    splen = []
    datas = []
    with open(hotpot) as tf:
        _json = json.load(tf)
        for _jsonline in _json:
            supporting_facts = _jsonline["supporting_facts"]
            question = _jsonline['question']
            context = _jsonline['context']
            context = {t: c for t, c in context}
            answer = _jsonline['answer']
            _id = _jsonline["_id"]

            passage = ""
            splen.append(len(supporting_facts))
            if len(supporting_facts) == 12:
                print(question)
                print(supporting_facts)
                print(json.dumps(context,indent=4))
                print(answer)
            for title, index in supporting_facts:
                if title in context and index < len(context[title]):
                    passage += title + ' ' + context[title][index]
            if answer not in ['yes', 'no'] and answer in passage:
                answer_start = passage.find(answer)
                count["normal"] += 1
            else:
                if answer in ['yes', 'no']:
                    answer_start = -1
                    count["YesNo"] += 1
                else:
                    continue

            data = {'title': '',
                    'paragraphs': [{
                        'qas': [{
                            'question': question,
                            'id': _id,
                            'answers': [{
                                'text': answer,
                                'answer_start': answer_start}],
                            'is_impossible': answer_start < 0}],
                        'context': passage}]
                    }
            datas.append(data)
    sd = {'version': "HotpotQA", 'data': datas}
    with open(squad, 'w') as sf:
        sf.write(json.dumps(sd) + '\n')
    print(count)
    print(collections.Counter(splen))


if __name__ == '__main__':
    hotpotToSquad(os.path.join(dataset_dir, train_file), os.path.join(dataset_dir, "squad/train.json"))
    hotpotToSquad(os.path.join(dataset_dir, dev_file), os.path.join(dataset_dir, "squad/dev.json"))
