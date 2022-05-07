import json
import os
import random
from datetime import datetime

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup, AdamW

from LongTextModels.dataloader.dataLoader import load_dataset
from LongTextModels.model.model import SentenceChoice
from LongTextModels.model.optimization import BERTAdam
from LongTextModels.tools.logger import get_logger
from LongTextModels.tools.squad import makeSquad


class Trainer(object):
    def __init__(self, hparams, mode='train'):
        self.hparams = hparams
        self.make_dir()  # 创建模型输出文件的文件夹
        self.log = get_logger(log_name="Trainer")
        self.mode = mode
        self.model = None
        self.dataset = None
        self.train_dataloader = None
        # 设置随机种子
        self.set_seed()
        self.summery_writer = self.init_SummaryWriter()
        self.pretrained_model_config, self.tokenizer = self.Load_pretrained_model_config()
        self.global_step, self.global_epoch, self.optimizer, self.scheduler = self.build_model()
        self.save_train_config()  # 保存该模型的训练超参数

    def Load_pretrained_model_config(self):
        """
        加载预训练模型参数 并加入针对该模型的自定义参数
        :return: pretrained_model_config, tokenizer
        """

        self.log.info("Load pretrained model from file %s ...", self.hparams.pretrainedModelPath)
        pretrained_model_config = AutoConfig.from_pretrained(self.hparams.pretrainedModelPath)
        tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrainedModelPath, use_fast=True)

        self.log.info("Load pretrained model config finished!!!")
        return pretrained_model_config, tokenizer

    def build_model(self):
        # Define model
        global_step = 0
        global_epoch = 0
        self.log.info("Define model...")

        # 创建模型
        self.model = SentenceChoice.from_pretrained(self.hparams.pretrainedModelPath, from_tf=False,
                                                    config=self.pretrained_model_config)
        # GPU or CPU
        self.log.info('use %s to train', self.hparams.device)
        self.model.to(self.hparams.device)

        # Use Multi-GPUs
        if len(self.hparams.gpu_ids) > 1 and self.hparams.device != 'cpu':
            self.model = nn.DataParallel(self.model, device_ids=self.hparams.gpu_ids)
            self.log.info("Use Multi-GPUs" + str(self.hparams.gpu_ids))
        else:
            self.log.info("Use 1 GPU")
        # TODO dataloader部分重写
        #  data

        dataset, examples, tokenizer = load_dataset(self.hparams, evaluate=False)
        self.dataset = dataset
        batch_size = self.hparams.per_gpu_batch_size * max(1, len(self.hparams.gpu_ids))
        train_sampler = RandomSampler(dataset)
        train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size, drop_last=True)
        self.train_dataloader = train_dataloader
        t_total = len(train_dataloader) // self.hparams.gradient_accumulation_steps * self.hparams.num_train_epochs

        """
        Adam
        # Define Loss and Optimizer
        # Prepare optimizer and schedule (linear warmup and decay)
        """
        # no_decay = ["bias", "LayerNorm.weight"]
        # optimizer_grouped_parameters = [
        #     {
        #         "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
        #         "weight_decay": 0.0,
        #     },
        #     {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
        #      "weight_decay": 0.0},
        # ]
        # optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=1e-8)
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        # )

        """
        BERTAdam
        # Define Loss and Optimizer
        # Create the learning rate scheduler.
        """
        no_decay = ['bias', 'gamma', 'beta']
        param_optimizer = list(self.model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if n not in no_decay], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if n in no_decay], 'weight_decay_rate': 0.0}
        ]
        optimizer = BERTAdam(optimizer_grouped_parameters,
                             lr=self.hparams.learning_rate,
                             warmup=self.hparams.warmup_proportion,
                             t_total=t_total)

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0.1 * t_total,  # Default value in run_glue.py
                                                    num_training_steps=t_total)

        # 加载已训练一部分的最优模型
        # if os.path.isfile(self.hparams.best_model_save_path) and self.hparams.load_half_model:
        #     checkpoint = torch.load(self.hparams.best_model_save_path)
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     self.log.info("Load optimizer from fine-turned checkpoint: '{}' (step {}, epoch {})"
        #                   .format(self.hparams.best_model_save_path, checkpoint['step'], checkpoint['epoch']))
        #     global_step = checkpoint['step']
        #     global_epoch = checkpoint['epoch'] + 1

        self.log.info("Define model finished!!!")

        return global_step, global_epoch, optimizer, scheduler

    def set_seed(self):
        random.seed(self.hparams.seed)
        np.random.seed(self.hparams.seed)
        torch.manual_seed(self.hparams.seed)
        if len(self.hparams.gpu_ids) > 0:
            torch.cuda.manual_seed_all(self.hparams.seed)

    def make_dir(self):
        if not os.path.exists(self.hparams.model_saved_path):
            os.makedirs(self.hparams.model_saved_path)
        _log_path = os.path.join(self.hparams.model_saved_path, 'logs')
        if not os.path.exists(_log_path):
            os.makedirs(_log_path)
        if not os.path.exists(self.hparams.tensorboard_path):
            os.makedirs(self.hparams.tensorboard_path)
        if not os.path.exists(self.hparams.eval_path):
            os.makedirs(self.hparams.eval_path)

    def init_SummaryWriter(self):
        """
        初始化 SummaryWriter
        :return: SummaryWriter
        """
        # today = str(datetime.today().month) + 'm' + str(
        #     datetime.today().day) + 'd_' + str(datetime.today().hour) + 'h-' + str(datetime.today().minute) + 'm'
        model_path = os.path.join(self.hparams.output_dir, self.hparams.current_model)
        writer_path = os.path.join(model_path, self.hparams.tensorboard_path)
        # exp_path = os.path.join(writer_path, 'exp_' + today)
        exp_path = writer_path
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        return SummaryWriter(exp_path)

    def save_train_config(self):
        """
        保存训练模型时的参数值
        :return:
        """

        self.log.info('model config output dir: {}'.format(self.hparams.model_config_file_path))
        model_config = {}
        for k, v in self.hparams.__dict__.items():
            if not str(k).__contains__('__') and str(k) != 'os' and str(k) != "make_dir":
                model_config[k] = v
        # 保存config
        json.dump(model_config, open(self.hparams.model_config_file_path, 'w'), indent=4)

    def save_best_model(self, model, optimizer, global_step, epoch):
        """
        保存训练过程中 性能最佳的模型
        :param model:
        :param optimizer:
        :param global_step:
        :param epoch:
        :return:
        """
        self.log.info("prepared to save the best model...")
        save_path = self.hparams.best_model_save_path  # 最优模型保存路径模型
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': global_step,
            'epoch': epoch
        }, save_path)
        self.log.info("best model have saved to " + save_path)

    def test(self, model=None, save_dir=None):
        self.log.info("***** Prepare for Test *****")
        self.log.info("Load dataset from file %s ...", self.hparams.testFile)
        dataset, examples, tokenizer = load_dataset(self.hparams, evaluate=True)
        batch_size = self.hparams.per_gpu_batch_size * max(1, len(self.hparams.gpu_ids))
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size)
        # if model is None:
        #     self.log.info("Load model from file %s ...", self.hparams.best_model_save_path)
        #     config = AutoConfig.from_pretrained(self.hparams.pretrainedModelPath)
        #     model = SentenceChoice.from_pretrained(self.hparams.best_model_save_path, from_tf=False, config=config)
        #     model.to(self.hparams.device)
        #     if len(self.hparams.gpu_ids) > 1 and not isinstance(model, torch.nn.DataParallel):
        #         model = torch.nn.DataParallel(model)

        self.log.info("***** Running test *****")
        self.log.info("Num examples = %d", len(dataset))
        self.log.info("Batch size = %d", batch_size)

        choiceList = []

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(self.hparams.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "question_id": batch[0],
                    "contexts_id": batch[1],
                    "syntactic_graph": batch[2],
                    "supporting_position": batch[3],
                }
                _, supporting_logits = model(**inputs)
            choice = supporting_logits[:, 1, :] > supporting_logits[:, 0, :]
            choice = choice.detach().cpu().tolist()
            supporting_position = batch[3].detach().cpu().tolist()
            for c, s in zip(choice, supporting_position):
                nc = [c[i] for i in range(len(c)) if s[2 * i + 1] != 0]
                choiceList.append(nc)

        self.log.info("Evaluation done.")

        choiceDict = {}
        right, wrong = 0, 0
        all_right, has_wrong = 0, 0
        data_s = []
        self.log.info("Evaluate EM and Compute new dataset.")
        for [_id, context_list, question, supporting_facts_list, answer], choice in tqdm(zip(examples, choiceList),
                                                                                         desc="Computing"):
            choiceDict[_id] = choice
            assert len(context_list) == len(choice), "Predict Length Maybe Wrong."
            _is_all_rigth = True
            new_context_list = []
            for context, fact, c in zip(context_list, supporting_facts_list, choice):
                if fact == c:
                    right += 1
                else:
                    wrong += 1
                    _is_all_rigth = False
                if c:
                    new_context_list.append(context)
            if _is_all_rigth:
                all_right += 1
            else:
                has_wrong += 1
            data_s.append(makeSquad(_id, new_context_list, question, answer))

        newSquadDataset = {'version': "HotpotQA", 'data': data_s}

        self.log.info("Save predictions.")
        # 新建文件夹
        out_path = os.path.join(self.hparams.eval_path, save_dir)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        output_choice_file = os.path.join(self.hparams.eval_path, save_dir, "choice.json")
        output_squad_file = os.path.join(self.hparams.eval_path, save_dir, "squad.json")

        with open(output_choice_file, 'w') as cf:
            json.dump(choiceDict, cf, ensure_ascii=False, indent=4)
        with open(output_squad_file, 'w') as sf:
            json.dump(newSquadDataset, sf, ensure_ascii=False, indent=4)

        self.log.info("Evaluate prediction.")
        results = {
            "acc": right / (right + wrong),
            "all_acc": all_right / (all_right + has_wrong),
        }
        self.log.info(results)
        result_file = os.path.join(out_path, "result.json")

        with open(result_file, 'w') as rf:
            json.dump(results, rf, ensure_ascii=False, indent=4)
        self.log.info("Save result.")
        return results

    def run_train_epoch(self, this_epoch, best_acc, train_begin_time):
        epoch = self.global_epoch + this_epoch
        global_step = self.global_step
        total_epoch = int(self.hparams.num_train_epochs)
        if self.hparams.do_train:
            bar_format = '{desc}{percentage:2.0f}%|{bar}|{n_fmt}/{total_fmt}[{elapsed}<{remaining}{postfix}]'
            epoch_iterator = tqdm(self.train_dataloader, ncols=120,
                                  bar_format=bar_format)
            epoch_iterator.set_description('Epoch: {}/{}'.format(epoch, total_epoch))  # 设置前缀 一般为epoch的信息
            # TODO train
            self.model.train()
            self.optimizer.zero_grad()  # reset gradient
            for step, batch in enumerate(epoch_iterator):
                batch = tuple(t.to(self.hparams.device) for t in batch)
                inputs = {
                    "question_id": batch[0],
                    "contexts_id": batch[1],
                    "syntactic_graph": batch[2],
                    "supporting_position": batch[3],
                    "supporting_fact_label": batch[4],
                }
                loss, _ = self.model(**inputs)
                # loss regularization
                if len(self.hparams.gpu_ids) > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
                if self.hparams.gradient_accumulation_steps > 1:
                    loss = loss / self.hparams.gradient_accumulation_steps
                # back propagation
                loss.backward()
                # update parameters of net
                # 累计一定step 再进行反向传播 梯度清零
                if (step + 1) % self.hparams.gradient_accumulation_steps == 0:
                    self.summery_writer.add_scalar('Train/Loss', loss.item(), global_step=global_step,
                                                   walltime=None)
                    # optimizer the net
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    # self.model.zero_grad() #
                    # 这里只清除 optimizer 添加到group中的参数梯度即可
                    self.optimizer.step()
                    self.scheduler.step()  # Update learning rate schedule
                    self.optimizer.zero_grad()  # reset gradient 清空过往梯度，为下一波梯度累加做准备
                    global_step += 1
                # epoch_iterator.set_postfix(loss=loss.item())
                UsedTime = "{}".format(str(datetime.utcnow() - train_begin_time).split('.')[0])
                Step = "{:6d}".format(step)
                Iter = "{:4d}".format(global_step)
                Loss = "{:7f}".format(loss.item())
                lr = "{:10f}".format(self.optimizer.param_groups[0]['lr'])
                epoch_iterator.set_postfix(UsedTime=UsedTime, Step=str(Step), Iter=str(Iter), Loss=str(Loss),
                                           lr=str(lr))
            self.global_step = global_step

        # evaluate
        if self.hparams.do_test:
            self.log.info("***** Running evaluation *****")
            self.model.eval()
            results = self.test(model=self.model, save_dir="epoch-{}".format(epoch))
            if results['acc'] > best_acc['acc']:
                best_acc = results
                best_acc['best_epoch'] = epoch
                # 保存最优模型
                self.save_best_model(model=self.model, optimizer=self.optimizer,
                                     global_step=global_step, epoch=epoch)
            return self.global_step, self.model, best_acc
        return self.global_step, self.model, 0
