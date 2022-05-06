import argparse
import json
import logging
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import AutoConfig, AdamW, get_linear_schedule_with_warmup

from squad import makeSquad
from model import SentenceChoice
from dataLoader import load_dataset

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args):
    logger.info("***** Prepare for Train *****")
    logger.info("Load dataset from file %s ...", os.path.join(args.datasetPath, args.trainFile))
    dataset, examples, tokenizer = load_dataset(args, evaluate=False)

    logger.info("Load model from file %s ...", args.modelPath)
    config = AutoConfig.from_pretrained(args.modelPath)
    model = SentenceChoice.from_pretrained(args.modelPath, from_tf=False, config=config)
    model.to(args.device)

    args.batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.batch_size)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running training *****")
    logger.info("Num examples = %d", len(dataset))
    logger.info("Num Epochs = %d", args.num_train_epochs)
    logger.info("Instantaneous batch size per GPU = %d", args.per_gpu_batch_size)
    logger.info("Total train batch size (w. parallel, distributed & accumulation) = %d", args.batch_size * args.gradient_accumulation_steps)
    logger.info("Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("Total optimization steps = %d", t_total)

    global_step = 1

    model.zero_grad()

    for epoch in range(int(args.num_train_epochs)):
        epoch_iterator = tqdm(train_dataloader, desc="Epoch " + str(epoch))
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "question_id": batch[0],
                "contexts_id": batch[1],
                "syntatic_graph": batch[2],
                "supporting_position": batch[3],
                "supporting_fact_label": batch[4],
            }

            loss, _ = model(**inputs)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            epoch_iterator.set_postfix(loss=loss.item())

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

        savePath = os.path.join(args.savePath, "epoch-{}".format(epoch))
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(savePath)
        tokenizer.save_pretrained(savePath)
        logger.info("Saving model checkpoint to %s", savePath)

        if args.do_test:
            test(args, model, savedir="epoch-{}".format(epoch))


def test(args, model=None, savedir=""):
    logger.info("***** Prepare for Test *****")
    logger.info("Load dataset from file %s ...", args.testFile)
    dataset, examples, tokenizer = load_dataset(args, evaluate=True)
    args.batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.batch_size)
    if model is None:
        logger.info("Load model from file %s ...", args.modelPath)
        config = AutoConfig.from_pretrained(args.modelPath)
        model = SentenceChoice.from_pretrained(args.modelPath, from_tf=False, config=config)
        model.to(args.device)
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

    logger.info("***** Running test *****")
    logger.info("Num examples = %d", len(dataset))
    logger.info("Batch size = %d", args.batch_size)

    choiceList = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "question_id": batch[0],
                "contexts_id": batch[1],
                "syntatic_graph": batch[2],
                "supporting_position": batch[3],
            }
            _, supporting_logits = model(**inputs)
        choice = supporting_logits[:, 1, :] > supporting_logits[:, 0, :]
        choice = choice.detach().cpu().tolist()
        supporting_position = batch[3].detach().cpu().tolist()
        for c, s in zip(choice, supporting_position):
            nc = [c[i] for i in range(len(c)) if s[2 * i + 1] != 0]
            choiceList.append(nc)

    logger.info("Evaluation done.")

    choiceDict = {}
    right, wrong = 0, 0
    all_right, has_wrong = 0, 0
    datas = []
    logger.info("Evaluate EM and Compute new dataset.")
    for [_id, context_list, question, supporting_facts_list, answer], choice in tqdm(zip(examples, choiceList), desc="Computing"):
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
        datas.append(makeSquad(_id, new_context_list, question, answer))

    newSquadDataset = {'version': "HotpotQA", 'data': datas}

    logger.info("Save predictions.")
    output_choice_file = os.path.join(args.savePath, savedir, "choice.json")
    output_squad_file = os.path.join(args.savePath, savedir, "squad.json")
    with open(output_choice_file, 'w') as cf:
        json.dump(choiceDict, cf, ensure_ascii=False, indent=4)
    with open(output_squad_file, 'w') as sf:
        json.dump(newSquadDataset, sf, ensure_ascii=False, indent=4)

    logger.info("Evaluate prediction.")
    results = {
        "acc": right / (right + wrong),
        "all_acc": all_right / (all_right + has_wrong),
    }
    logger.info(results)
    result_file = os.path.join(args.savePath, savedir, "results")
    with open(result_file, 'w') as rf:
        json.dump(results, rf, ensure_ascii=False, indent=4)
    logger.info("Save result.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action="store_true")
    parser.add_argument('--do_test', action="store_true")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--datasetPath', type=str, required=True)
    parser.add_argument('--trainFile', type=str)
    parser.add_argument('--testFile', type=str)
    parser.add_argument('--modelPath', type=str, required=True)
    parser.add_argument('--savePath', type=str, required=True)
    parser.add_argument('--tempPath', type=str, required=True)
    parser.add_argument("--per_gpu_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--num_train_epochs", default=4, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--learning_rate", default=1e-2, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_steps", default=1000, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.",
                        )
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.",
                        )
    parser.add_argument("--seed", type=int, default=7455100, help="random seed for initialization")
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )

    if not os.path.isdir(args.datasetPath):
        exit("datasetPath IS NOT A DIRCTIONARY. " + args.datasetPath)
    if not os.path.isdir(args.modelPath):
        exit("modelPath IS NOT A DIRCTIONARY. " + args.modelPath)
    if not os.path.isdir(args.tempPath):
        exit("tempPath IS NOT A DIRCTIONARY. " + args.tempPath)

    testFile = os.path.join(args.datasetPath, args.testFile)
    if not os.path.isfile(testFile):
        exit("There is no testFile OR testFile is not EXIST. " + testFile)

    if args.do_train:
        if not os.path.isfile(os.path.join(args.datasetPath, args.trainFile)):
            exit("There is no trainFile OR trainFile is not EXIST. " + os.path.join(args.datasetPath, args.trainFile))
        train(args)
    elif args.do_test:
        test(args)
