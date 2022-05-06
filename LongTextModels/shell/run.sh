/data2/maqi/LongTextDatasets/LongTextModels/main/main.py

nohup python -m LongTextModels.main.main.py >> log.txt 2>&1 &

tensorboard --logdir=./ --port 8124 --bind_all
nohup tensorboard --logdir=./ --port 8124 --bind_all >> log.txt 2>&1 &