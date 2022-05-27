import copy
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
from LongTextModels.tools.logger import get_logger
from LongTextModels.tools.squad import makeSquad


class Trainer(object):
    def __init__(self, hparams, mode='train'):
        self.hparams = hparams
        self.make_dir()  # 创建模型输出文件的文件夹
        self.log = get_logger(log_name="Trainer")
        self.mode = mode
        self.model = None
        self.eval_dataset = None
        self.eval_examples = None
        self.train_dataset = None
        self.train_dataloader = None
        self.eval_dataloader = None
        self.batch_size = self.hparams.per_gpu_batch_size * max(1, len(self.hparams.gpu_ids))
        self.set_seed()  # 设置随机种子
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

        # TODO dataloader部分重写
        #  data

        # 训练集
        self.log.info("Load dev dataset from file %s ...", self.hparams.trainFile)
        self.train_dataset, train_examples, tokenizer = load_dataset(self.hparams, mode="train")
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.batch_size,
                                      drop_last=True)
        self.train_dataloader = train_dataloader
        t_total = len(train_dataloader) // self.hparams.gradient_accumulation_steps * self.hparams.num_train_epochs
        # 验证集
        self.log.info("Load dev dataset from file %s ...", self.hparams.devFile)
        self.eval_dataset, self.eval_examples, tokenizer = load_dataset(self.hparams, mode="eval")
        eval_sampler = SequentialSampler(self.eval_dataset)
        self.eval_dataloader = DataLoader(self.eval_dataset, sampler=eval_sampler, batch_size=self.batch_size,
                                          drop_last=True)

        """
        Adam
        # Define Loss and Optimizer
        # Prepare optimizer and schedule (linear warmup and decay)
        """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 1e-2,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0}
        ]

        # optimizer_grouped_parameters = [
        #     {"params": [p for n, p in self.model.named_parameters() if n.startswith("RELoss_fn")],
        #      "weight_decay": 0.0},
        #     # {
        #     #     "params": [p for n, p in self.model.named_parameters() if not n.startswith("RELoss_fn") if
        #     #                not any(nd in n for nd in no_decay)],
        #     #     "weight_decay": 0.0,
        #     # },
        #     # {"params": [p for n, p in self.model.named_parameters() if not n.startswith("RELoss_fn") if
        #     #             any(nd in n for nd in no_decay)],
        #     #  "weight_decay": 0.0},
        # ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=1e-8,
                          correct_bias=False)  # 要重现BertAdam特定的行为，请设置correct_bias = False
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_proportion * t_total, num_training_steps=t_total
        )
        # num_warmup_steps=0.1*t_total 表示全部训练步骤的前warmup_proportion %，在这一阶段，学习率线性增加；此后，学习率线性衰减。

        # 加载已训练一部分的最优模型
        if os.path.isfile(self.hparams.best_model_save_path) and self.hparams.load_part_model:
            self.log.info('Load part model from :' + self.hparams.best_model_save_path)
            checkpoint = torch.load(self.hparams.best_model_save_path)
            optimizer.load_state_dict(checkpoint['optimizer'])
            self.log.info("Load optimizer from fine-turned checkpoint: '{}' (step {}, epoch {})"
                          .format(self.hparams.best_model_save_path, checkpoint['step'], checkpoint['epoch']))
            global_step = checkpoint['step']
            global_epoch = checkpoint['epoch'] + 1
            self.log.info("Load model finished!!!")

        # GPU or CPU
        self.log.info('use %s to train', self.hparams.device)
        self.model.to(self.hparams.device)

        # Use Multi-GPUs
        if len(self.hparams.gpu_ids) > 1 and self.hparams.device != 'cpu':
            self.model = nn.DataParallel(self.model, device_ids=self.hparams.gpu_ids)
            self.log.info("Use Multi-GPUs" + str(self.hparams.gpu_ids))
        else:
            self.log.info("Use 1 GPU")

        self.log.info("Define model finished!!!")
        self.model.zero_grad()

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
        dataset, examples, tokenizer = load_dataset(self.hparams, mode=True)
        batch_size = self.hparams.per_gpu_batch_size * max(1, len(self.hparams.gpu_ids))
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size)
        if model is None:
            self.log.info("Load model from file %s ...", self.hparams.best_model_save_path)
            config = AutoConfig.from_pretrained(self.hparams.pretrainedModelPath)
            model = SentenceChoice.from_pretrained(self.hparams.best_model_save_path, from_tf=False, config=config)
            model.to(self.hparams.device)
            if len(self.hparams.gpu_ids) > 1 and not isinstance(model, torch.nn.DataParallel):
                model = torch.nn.DataParallel(model)

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
            if self.hparams.testFunction == '0':
                choice = supporting_logits[:, 1, :] > supporting_logits[:, 0, :]
                choice = choice.detach().cpu().tolist()
                supporting_position = batch[3].detach().cpu().tolist()
                for c, s in zip(choice, supporting_position):
                    nc = [c[i] for i in range(len(c)) if s[2 * i + 1] != 0]
                    choiceList.append(nc)
            elif self.hparams.testFunction == '1':
                supporting_logits = supporting_logits.softmax(dim=1)
                choice_logits = supporting_logits[:, 1, :].detach().cpu().tolist()  # 预测为正例的概率
                supporting_position = batch[3].detach().cpu().tolist()
                for c, s in zip(choice_logits, supporting_position):
                    nc = [c[i] for i in range(len(c)) if s[2 * i + 1] != 0]
                    choiceList.append(nc)
            else:
                self.log.info("Error testFunction {}.".format(self.hparams.testFunction))
                return

        self.log.info("Evaluation done.")

        choiceDict = {}
        choiceDict_PR = {}
        right, wrong = 0, 0
        all_right, has_wrong = 0, 0
        datas = []
        self.log.info("Evaluate EM and Compute new dataset.")
        for [_id, context_list, question, supporting_facts_list, answer], choice in tqdm(zip(examples, choiceList),
                                                                                         desc="Computing"):
            if len(choice) == 0:
                continue
            temp_choice = copy.deepcopy(choice)
            # choiceDict[_id] = choice # 概率
            assert len(context_list) == len(choice), "Predict Length Maybe Wrong."
            if self.hparams.testFunction == '0':
                _is_all_right = True
                new_context_list = []
                for context, fact, c in zip(context_list, supporting_facts_list, choice):
                    if fact == c:
                        right += 1
                    else:
                        wrong += 1
                        _is_all_right = False
                    if c:
                        new_context_list.append(context)
                if _is_all_right:
                    all_right += 1
                else:
                    has_wrong += 1
            elif self.hparams.testFunction == '1':
                _is_right = []
                new_context_list = []
                choose_best = sorted(range(len(choice)), key=choice.__getitem__, reverse=True)[:self.hparams.bestN]
                bottom = choice[choose_best[-1]]
                temp_choice = [True if i >= bottom else False for i in choice]
                for cb in choose_best:
                    new_context_list.append(context_list[cb])
                    _right = supporting_facts_list[cb]
                    _is_right.append(_right)
                fact_count = supporting_facts_list.count(1)
                right_count = _is_right.count(1)
                if right_count == fact_count or right_count == self.hparams.bestN:
                    all_right += 1
                    right += self.hparams.bestN
                else:
                    has_wrong += 1
                    _wrong = min(fact_count - right_count, self.hparams.bestN - right_count)
                    wrong += _wrong
                    right += self.hparams.bestN - _wrong
            else:
                self.log.info("Error testFunction {}.".format(self.hparams.testFunction))
                return
            choiceDict[_id] = temp_choice
            choiceDict_PR[_id] = choice
            datas.append(makeSquad(_id, new_context_list, question, answer))

        newSquadDataset = {'version': "HotpotQA", 'data': datas}
        self.log.info("Save predictions.")
        # 新建文件夹
        out_path = os.path.join(self.hparams.eval_path, save_dir)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        output_choice_file = os.path.join(self.hparams.eval_path, save_dir, "choice.json")
        output_choice_PR_file = os.path.join(self.hparams.eval_path, save_dir, "PR.json")
        output_squad_file = os.path.join(self.hparams.eval_path, save_dir, "squad.json")

        with open(output_choice_file, 'w') as cf:
            json.dump(choiceDict, cf, ensure_ascii=False, indent=4)
        with open(output_choice_PR_file, 'w') as cf:
            json.dump(choiceDict_PR, cf, ensure_ascii=False, indent=4)
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

    def eval(self, save_dir=None):
        self.log.info("***** Prepare for Eval *****")
        self.log.info("Load dataset from file %s ...", self.hparams.devFile)

        self.log.info("***** Running eval *****")
        self.log.info("Num examples = %d", len(self.eval_dataset))
        self.log.info("Batch size = %d", self.batch_size)
        self.model.eval()
        choiceList = []

        for step, batch in enumerate(tqdm(self.eval_dataloader, desc="Evaluating")):

            batch = tuple(t.to(self.hparams.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "question_id": batch[0],
                    "contexts_id": batch[1],
                    "syntactic_graph": batch[2],
                    "supporting_position": batch[3],
                    "supporting_fact_label": batch[4],
                }

                re_loss, eval_loss, focal_loss, supporting_logits = self.model(**inputs)
                if len(self.hparams.gpu_ids) > 1:
                    re_loss = re_loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
                    eval_loss = eval_loss.mean()
                    focal_loss = focal_loss.mean()

                # summery_writer
                self.summery_writer.add_scalar('Val/re_loss/' + str(save_dir), re_loss.item(), global_step=step,
                                               walltime=None)
                self.summery_writer.add_scalar('Val/loss/' + str(save_dir), eval_loss.item(), global_step=step,
                                               walltime=None)
                self.summery_writer.add_scalar('Val/focal_loss/' + str(save_dir), focal_loss.item(), global_step=step,
                                               walltime=None)

            if self.hparams.testFunction == '0':
                choice = supporting_logits[:, 1, :] > supporting_logits[:, 0, :]
                choice = choice.detach().cpu().tolist()
                supporting_position = batch[3].detach().cpu().tolist()
                for c, s in zip(choice, supporting_position):
                    nc = [c[i] for i in range(len(c)) if s[2 * i + 1] != 0]
                    choiceList.append(nc)
            elif self.hparams.testFunction == '1':
                supporting_logits = supporting_logits.softmax(dim=1)
                choice_logits = supporting_logits[:, 1, :].detach().cpu().tolist()  # 预测为正例的概率
                supporting_position = batch[3].detach().cpu().tolist()
                for c, s in zip(choice_logits, supporting_position):
                    nc = [c[i] for i in range(len(c)) if s[2 * i + 1] != 0]
                    choiceList.append(nc)
            else:
                self.log.info("Error testFunction {}.".format(self.hparams.testFunction))
                return

        self.log.info("Evaluation done.")

        choiceDict = {}
        choiceDict_PR = {}
        right, wrong = 0, 0
        all_right, has_wrong = 0, 0
        datas = []
        self.log.info("Evaluate EM and Compute new dataset.")
        for [_id, context_list, question, supporting_facts_list, answer], choice in tqdm(
                zip(self.eval_examples, choiceList),
                desc="Computing"):
            if len(choice) == 0:
                continue
            temp_choice = copy.deepcopy(choice)
            # choiceDict[_id] = choice # 概率
            assert len(context_list) == len(choice), "Predict Length Maybe Wrong."
            if self.hparams.testFunction == '0':
                _is_all_right = True
                new_context_list = []
                for context, fact, c in zip(context_list, supporting_facts_list, choice):
                    if fact == c:
                        right += 1
                    else:
                        wrong += 1
                        _is_all_right = False
                    if c:
                        new_context_list.append(context)
                if _is_all_right:
                    all_right += 1
                else:
                    has_wrong += 1
            elif self.hparams.testFunction == '1':
                _is_right = []
                new_context_list = []
                choose_best = sorted(range(len(choice)), key=choice.__getitem__, reverse=True)[:self.hparams.bestN]
                bottom = choice[choose_best[-1]]
                temp_choice = [True if i >= bottom else False for i in choice]
                for cb in choose_best:
                    new_context_list.append(context_list[cb])
                    _right = supporting_facts_list[cb]
                    _is_right.append(_right)
                fact_count = supporting_facts_list.count(1)
                right_count = _is_right.count(1)
                if right_count == fact_count or right_count == self.hparams.bestN:
                    all_right += 1
                    right += self.hparams.bestN
                else:
                    has_wrong += 1
                    _wrong = min(fact_count - right_count, self.hparams.bestN - right_count)
                    wrong += _wrong
                    right += self.hparams.bestN - _wrong
            else:
                self.log.info("Error testFunction {}.".format(self.hparams.testFunction))
                return
            choiceDict[_id] = temp_choice
            choiceDict_PR[_id] = choice
            datas.append(makeSquad(_id, new_context_list, question, answer))

        newSquadDataset = {'version': "HotpotQA", 'data': datas}
        self.log.info("Save predictions.")
        # 新建文件夹
        out_path = os.path.join(self.hparams.eval_path, save_dir)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        output_choice_file = os.path.join(self.hparams.eval_path, save_dir, "choice.json")
        output_choice_PR_file = os.path.join(self.hparams.eval_path, save_dir, "PR.json")
        output_squad_file = os.path.join(self.hparams.eval_path, save_dir, "squad.json")

        with open(output_choice_file, 'w') as cf:
            json.dump(choiceDict, cf, ensure_ascii=False, indent=4)
        with open(output_choice_PR_file, 'w') as cf:
            json.dump(choiceDict_PR, cf, ensure_ascii=False, indent=4)
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
        self.log.info("Save eval result.")
        self.model.train()
        return results

    def run_train_epoch(self, this_epoch, best_acc, train_begin_time):
        self.log.info("***** Running train *****")
        self.log.info("Num examples = %d", len(self.train_dataset))
        self.log.info("Batch size = %d", self.batch_size)
        epoch = self.global_epoch + this_epoch
        global_step = self.global_step
        total_epoch = int(self.hparams.num_train_epochs)
        if self.hparams.do_train:
            bar_format = '{desc}{percentage:2.0f}%|{bar}|{n_fmt}/{total_fmt}[{elapsed}<{remaining}{postfix}]'
            epoch_iterator = tqdm(self.train_dataloader, ncols=150,
                                  bar_format=bar_format)
            epoch_iterator.set_description('Epoch: {}/{}'.format(epoch, total_epoch))  # 设置前缀 一般为epoch的信息
            # TODO train
            self.model.train()
            running_loss, count = 0.0, 0
            # self.optimizer.zero_grad()  # reset gradient
            for step, batch in enumerate(epoch_iterator):
                batch = tuple(t.to(self.hparams.device) for t in batch)
                inputs = {
                    "question_id": batch[0],
                    "contexts_id": batch[1],
                    "syntactic_graph": batch[2],
                    "supporting_position": batch[3],
                    "supporting_fact_label": batch[4],
                }
                # b_batch = batch[4].tolist()
                # for b in b_batch:
                #     sum1 = [i for i in b if i == 1]
                #     sum0 = [i for i in b if i == 0]
                #     sum_len.append(len(sum1) / len(sum0))
                re_loss, loss, focal_loss, _ = self.model(**inputs)
                # loss regularization
                if len(self.hparams.gpu_ids) > 1:
                    loss = loss.mean()
                    re_loss = re_loss.mean()
                    focal_loss = focal_loss.mean()
                    # mean() to average on multi-gpu parallel (not distributed) training
                if self.hparams.gradient_accumulation_steps > 1:
                    loss = loss / self.hparams.gradient_accumulation_steps
                    re_loss = re_loss / self.hparams.gradient_accumulation_steps
                    focal_loss = focal_loss / self.hparams.gradient_accumulation_steps
                # back propagation
                # loss.backward()
                # running_loss += loss.item()

                # re_loss.backward()
                # running_loss += re_loss.item()

                focal_loss.backward()
                running_loss += focal_loss.item()

                # update parameters of net
                # 累计一定step 再进行反向传播 梯度清零
                if (step + 1) % self.hparams.gradient_accumulation_steps == 0:
                    self.summery_writer.add_scalar('Train/Loss', loss.item(), global_step=global_step,
                                                   walltime=None)
                    # TODO 暂时不能加梯度裁剪
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                    #                                max_norm=1.0)  # 梯度裁剪不再在AdamW中了(因此你可以毫无问题地使用放大器)

                    # optimizer the net
                    self.optimizer.step()
                    self.scheduler.step()  # Update learning rate schedule
                    # self.model.zero_grad()
                    # 这里只清除 optimizer 添加到group中的参数梯度即可
                    self.optimizer.zero_grad()  # reset gradient 清空过往梯度，为下一波梯度累加做准备
                    global_step += 1
                    count += 1

                # 迭代global_step轮次之后 记录 平均loss
                if global_step % 200 == 0 and count != 0:
                    self.log.info("step: {}, loss: {:.3f}".format(global_step, running_loss / count))
                    # summery_writer
                    self.summery_writer.add_scalar('Train/running_loss', running_loss / count, global_step=global_step,
                                                   walltime=None)
                    running_loss, count = 0.0, 0
                # 更新进度条
                UsedTime = "{}".format(str(datetime.utcnow() - train_begin_time).split('.')[0])
                Step = "{:6d}".format(step)
                Iter = "{:4d}".format(global_step)
                Loss = "{:7f}".format(loss.item())
                RELoss = "{:7f}".format(re_loss.item())
                Focal_loss = "{:7f}".format(focal_loss.item())
                lr = "{:10f}".format(self.optimizer.param_groups[0]['lr'])
                epoch_iterator.set_postfix(UsedTime=UsedTime, Step=str(Step), Iter=str(Iter), Loss=str(Loss),
                                           RELoss=str(RELoss), Focal_loss=str(Focal_loss), lr=str(lr))
            self.global_step = global_step

        if self.hparams.do_eval:
            self.log.info("***** Running evaluation *****")
            results = self.eval(save_dir="eval-epoch-{}".format(epoch))
            if results['all_acc'] >= best_acc['all_acc']:
                best_acc = results
                best_acc['best_epoch'] = epoch
                # 保存最优模型
                self.save_best_model(model=self.model, optimizer=self.optimizer,
                                     global_step=global_step, epoch=epoch)
            return self.global_step, self.model, best_acc
        return self.global_step, self.model, 0
