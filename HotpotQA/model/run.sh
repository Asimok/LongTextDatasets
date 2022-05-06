export CUDA_VISIBLE_DEVICES=0,1,2,3

python run.py \
    --do_train \
    --do_test \
    --datasetPath /data2/wangbingchao/dataset/HotpotQA/ \
    --trainFile hotpot_train_v1.1.json \
    --testFile hotpot_dev_distractor_v1.json \
    --modelPath /data2/wangbingchao/database/bert_pretrained/bert-base-uncased \
    --tempPath /data2/wangbingchao/temp/ \
    --savePath /data2/wangbingchao/output/Hotpot_sc


