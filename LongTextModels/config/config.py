"""
模型超参数及资源路径
"""
import os


# 创建文件夹
def make_dir():
    print('创建文件夹...')
    if not os.path.exists(model_saved_path):
        os.makedirs(model_saved_path)
    _log_path = os.path.join(model_saved_path, 'logs')
    if not os.path.exists(_log_path):
        os.makedirs(_log_path)
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)


# 资源路径
# wbc
datasetPath = '/data2/wangbingchao/dataset/HotpotQA/'
trainFile = 'hotpot_train_v1.1.json'
devFile = 'hotpot_dev_fullwiki_v1.json'
testFile = 'hotpot_dev_fullwiki_v1.json'
# maqi
# datasetPath = '/data2/maqi/LongTextDatasets/HotpotQA_datasets/'
# trainFile = 'simplify_hotpot_train_v1.1.json'
# devFile = 'simplify_hotpot_dev_fullwiki_v1.json'
# testFile = 'simplify_hotpot_dev_fullwiki_v1.json'

pretrainedModelPath = '/data2/wangbingchao/database/bert_pretrained/bert-base-uncased'
cachePath = '/data2/maqi/LongTextDatasets/LongTextModels/cache'  # 预处理数据的缓存

# 存储路径
output_dir = '/data2/maqi/LongTextDatasets/LongTextModels/output/'
current_model = 'learning_rate'  # 不同模型的日志保存目录
model_saved_path = output_dir + current_model  # 当前训练模型保存路径
log_path = model_saved_path + '/logs/log.txt'  # 日志保存在当前训练的模型文件夹下
tensorboard_path = model_saved_path + '/tensorboard_runs'  # output_dir + current_model + tensorboard_path + date
best_model_save_path = model_saved_path + '/best_model_checkpoint.pth.tar'  # modelSavePath + best_model_save_path
eval_path = model_saved_path + '/results'
model_config_file_path = model_saved_path + '/config.txt'  # output_dir + current_model + model_config_file_path
# 创建文件夹移至 模型中创建文件夹
make_dir()

# GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
gpu_ids = [0, 1, 2, 3]  # 注意：在程序中设置gpu_id仍要从0开始，gpu_ids为 CUDA_VISIBLE_DEVICES 的索引
device = "cuda"

# main
do_train = True
do_eval = True
overwrite_cache = False
load_part_model = False  # 加载已训练一部分的最优模型

# train
per_gpu_batch_size = 12  # 每个gpu上的batch
num_train_epochs = 10
learning_rate = 1e-3
warmup_steps = 100
warmup_proportion = 0.05
gradient_accumulation_steps = 1  # 这个操作就相当于将batch_size扩大了gradient_accumulate_steps倍
true_loss_proportion = 0.5
testFunction = '1'
bestN = 4
max_query_length = 64
seed = 703
