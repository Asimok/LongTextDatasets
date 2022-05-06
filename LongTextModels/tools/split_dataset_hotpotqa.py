import json

read_dataset_path = '/data2/wangbingchao/dataset/HotpotQA/hotpot_train_v1.1.json'
output_dataset_path_train = '/data2/maqi/LongTextDatasets/HotpotQA_datasets/simplify_hotpot_train_v1.1.json'

data = json.load(open(read_dataset_path, 'r'))
out_data = data[0:100]
json.dump(out_data, open(output_dataset_path_train, 'w'))
print('saved simplify_hotpot_train_v1.1.json!')
# dev
read_dataset_path_dev = '/data2/wangbingchao/dataset/HotpotQA/hotpot_dev_fullwiki_v1.json'
output_dataset_path_dev = '/data2/maqi/LongTextDatasets/HotpotQA_datasets/simplify_hotpot_dev_fullwiki_v1.json'
data = json.load(open(read_dataset_path_dev, 'r'))
out_data = data[0:100]
json.dump(out_data, open(output_dataset_path_dev, 'w'))
print('saved simplify_hotpot_dev_fullwiki_v1.json')
