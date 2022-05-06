from tqdm import tqdm

data_path = '/data2/maqi/datasets/NQ/rawNQ/v1.0-simplified-nq-train.jsonl'
out_dataset_path_chunk1 = '/data2/maqi/datasets/NQ/rawNQ/train_chunks/chunk1_v1.0-simplified-nq-train.jsonl'
out_dataset_path_chunk2 = '/data2/maqi/datasets/NQ/rawNQ/train_chunks/chunk2_v1.0-simplified-nq-train.jsonl'
out_dataset_path_chunk3 = '/data2/maqi/datasets/NQ/rawNQ/train_chunks/chunk3_v1.0-simplified-nq-train.jsonl'
out_dataset_path_chunk4 = '/data2/maqi/datasets/NQ/rawNQ/train_chunks/chunk4_v1.0-simplified-nq-train.jsonl'


# less_dataset_path = '/data2/maqi/datasets/NQ/less_v1.0-simplified-nq-train.jsonl'

# data_path = '/data2/maqi/datasets/NQ/test-clean-nq-train.jsonl'

# data_path = '/data2/maqi/datasets/NQ/cleanNQ/clean-nq-dev.jsonl'
# less_dataset_path = '/data2/maqi/datasets/NQ/less_clean-nq-dev.jsonl'
def save(out_path, chunk):
    with open(out_path, 'w') as fo:
        for line in tqdm(chunk):
            fo.writelines(line)
    print('saved ', out_path)


with open(data_path, 'r') as f:
    data_set = f.readlines()
len_data_set = len(data_set)
len_chunk1 = int(len_data_set * 0.25)
len_chunk2 = int(len_data_set * 0.5)
len_chunk3 = int(len_data_set * 0.75)
print(len_data_set, len_chunk1, len_chunk2 - len_chunk1, len_chunk3 - len_chunk2, len_data_set - len_chunk3)

chunk1 = data_set[0:len_chunk1]
save(out_dataset_path_chunk1, chunk1)
chunk2 = data_set[len_chunk1:len_chunk2]
save(out_dataset_path_chunk2, chunk2)
chunk3 = data_set[len_chunk2:len_chunk3]
save(out_dataset_path_chunk3, chunk3)
chunk4 = data_set[len_chunk3:]
save(out_dataset_path_chunk4, chunk4)

print(len_data_set, len_chunk1, len_chunk2 - len_chunk1, len_chunk3 - len_chunk2, len_data_set - len_chunk3)
