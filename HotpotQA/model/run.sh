export CUDA_VISIBLE_DEVICES=4,5,6,7

python run.py \
  --do_train \
  --do_test \
  --num_train_epochs 12 \
  --testFunction 1 \
  --per_gpu_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 3e-5 \
  --datasetPath /data2/wangbingchao/dataset/HotpotQA/ \
  --trainFile hotpot_train_v1.1.json \
  --testFile hotpot_dev_distractor_v1.json \
  --modelPath /data2/wangbingchao/database/bert_pretrained/bert-base-uncased \
  --tempPath /data2/wangbingchao/temp/ \
  --savePath /data2/wangbingchao/output/Hotpot_sc
