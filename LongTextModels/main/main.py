import json
import os
import sys
from datetime import datetime

from LongTextModels.config import config
from LongTextModels.model.trainer import Trainer

sys.path.append('/data2/maqi/LongTextDatasets/')
hparams = config
trainer = Trainer(hparams=hparams, mode='train')
best_acc = {"acc": 0, "all_acc": 0}
train_begin = datetime.utcnow()  # Times
for epoch in range(int(hparams.num_train_epochs)):
    global_step, model, best_acc = trainer.run_train_epoch(this_epoch=epoch + 1, best_acc=best_acc,
                                                           train_begin_time=train_begin)
    print("```````````````````````````")
    print("epoch: {}, global_step: {}".format((epoch + 1), global_step))
    print("best_acc : ", best_acc)
    print("```````````````````````````")
    if not hparams.do_train:
        print('finished eval!!!')
        break

best_result_file = os.path.join(hparams.eval_path, "best_result.json")

with open(best_result_file, 'w') as rf:
    json.dump(best_acc, rf, ensure_ascii=False, indent=4)
