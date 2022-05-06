# train
nohup python clean_NQ_new.py train_chunk1 >> train_chunk1.log 2>&1 &
nohup python clean_NQ_new.py train_chunk2 >> train_chunk2.log 2>&1 &
nohup python clean_NQ_new.py train_chunk3 >> train_chunk3.log 2>&1 &
nohup python clean_NQ_new.py train_chunk4 >> train_chunk4.log 2>&1 &
# dev
nohup python clean_NQ_new.py dev >> dev.log 2>&1 &
# test
nohup python clean_NQ_new.py test >> test.log 2>&1 &

#分块
nohup python split_NQ.py >> split_NQ.log 2>&1 &