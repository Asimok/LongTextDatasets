import os

import torch

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# gpu = torch.cuda.device_count()
# print('gpu')
# print(gpu)
from LongTextModels.config import config
from LongTextModels.dataloader.dataLoader import load_dataset
args =config
dataset, examples, tokenizer = load_dataset(config=config, evaluate=False)