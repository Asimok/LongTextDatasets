import bisect
import json
import os

import spacy
import torch
from spacy.tokens import Doc
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, is_torch_available
from LongTextModels.tools.logger import get_logger

logger = get_logger(log_name="DropDataloader")
spacy_en = spacy.load('en_core_web_sm')
spacy_en.disable_pipes('tagger', 'attribute_ruler', 'attribute_ruler', 'lemmatizer', 'ner')


def get_examples(datasetFile: str):
    """
    :return:
         [[_id, context_list, question, supporting_facts_list, answer],...]
    """
    if not datasetFile:
        logger.error("DatasetFile need a datasetFile to pass in")
    examples = []
    with open(datasetFile, 'r') as df:
        _json = json.load(df)
        for _jsonline in _json:
            supporting_facts = _jsonline["supporting_facts"]
            question = _jsonline['question']
            context = _jsonline['context']
            answer = _jsonline['answer']
            supporting_facts_dict = {}
            _id = _jsonline["_id"]
            for _title, _index in supporting_facts:
                if _title not in supporting_facts_dict:
                    supporting_facts_dict[_title] = {_index}
                else:
                    supporting_facts_dict[_title].add(_index)
            supporting_facts_list = []
            context_list = []
            for _title, _context in context:
                for _index, _sentence in enumerate(_context):
                    context_list.append(_title + '. ' + _sentence.strip())
                    supporting_facts = False
                    for k, v in supporting_facts_dict.items():
                        if (_title in k or k in _title) and _index in v:
                            supporting_facts = True
                    supporting_facts_list.append(int(supporting_facts))
            examples.append([_id, context_list, question, supporting_facts_list, answer])
    return examples


def get_spacy_tree_0(sentence: str, offset: list):
    """
    :return:
         use auto spacy to get a tree and
         if many tokens make up a word,
         let these token same as the word.
    """
    m = []
    words = []
    doc = spacy_en(sentence)
    index = 0
    for token in doc:
        t = str(token)
        index = sentence.find(t, index)
        assert index != -1, 'token: {} not found in sentence: {}'.format(t, sentence)
        index += len(t)
        words.append(bisect.bisect_left(offset, (index, index)) - 1)
        m.append(token.head.i)
    assert len(offset) == words[-1] + 1, 'need length: {}\nonly has: {}'.format(offset, words[-1] + 1)
    newm = []
    j = 0
    for i in range(len(offset)):
        if words[j] < i:
            j += 1
        newm.append(m[j])
    return newm


def get_spacy_tree_1(tokens: list):
    """
    :return:
         use spacy.tokens.Doc to get a tree and
         '##' may in tokens and
         let these token same as words.
    """
    m = []
    doc = Doc(spacy_en.vocab, tokens)
    for name, tool in spacy_en.pipeline:
        tool(doc)
    for token in doc:
        m.append(token.head.i)
    return m


# def miniGraphToBigGraph(trees: list, length: int) -> [[], ]:
#     newGraph = [[0 for _ in range(length)] for _ in range(length)]
#     p = 0
#     for tree in trees:
#         l = len(tree)
#         for i, t in enumerate(tree):
#             newGraph[p + i][p + i] = 1
#             newGraph[p + i][p + t] = 1
#             newGraph[p + t][p + i] = 1
#         p += l
#     return newGraph


def miniGraphToSmallGraph(trees: list, length: int) -> list:
    graph = []
    i = 0
    for tree in trees:
        graph.extend([i + t for t in tree])
        i += len(tree)
    return graph + (length - len(graph)) * [length - 1]


def load_dataset(config, mode="train"):
    """
    :mode: train dev test
    :return: dataset, examples, context_offsets, tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(config.pretrainedModelPath, use_fast=True)
    temp_file = "cache_{}_{}_{}".format(
        mode,
        config.testFile if mode else config.trainFile,
        config.pretrainedModelPath.strip('/').split("/")[-1],
    )
    temp_file = os.path.join(config.cachePath, temp_file)
    if os.path.exists(temp_file) and not config.overwrite_cache:
        logger.info("Loading dataset from cached file %s", temp_file)
        dataset_and_examples = torch.load(temp_file)
        dataset, examples = dataset_and_examples["dataset"], dataset_and_examples["examples"]
        return dataset, examples, tokenizer
    else:
        logger.info("Creating dataset ...")

    examples = get_examples(os.path.join(config.datasetPath, config.testFile if mode else config.trainFile))
    # maybe use 10 passages can divide in to different parts and use BERT
    question_ids = []
    contexts_ids = []
    syntactic_graphs = []
    supporting_positions = []
    supporting_fact_labels = []
    question_max_length = 0
    contexts_max_length = 0
    supporting_max_length = 0
    for _, context_list, question, supporting_facts_list, _ in tqdm(examples, desc='Tokenizer'):
        syntactic_tree = []
        supporting_position = []
        question_token = tokenizer(question, add_special_tokens=False)
        question_id = question_token[0].ids
        # question_id = tokenizer.encode(question, truncation=True, max_length=args.max_query_length)
        # syntactic_tree.append(get_spacy_tree_0(question, question_token[0].offsets))
        # syntactic_tree.append(get_spacy_tree_1(question_token[0].tokens))
        q_len, c_len = len(question_id), 0
        question_ids.append(question_id)
        contexts_id = []
        for context in context_list:
            supporting_position.append(c_len)
            context_token = tokenizer(context, add_special_tokens=True)
            context_id = context_token[0].ids
            syntactic_tree.append([1])
            syntactic_tree.append(get_spacy_tree_0(context, context_token[0].offsets[1:-1]))
            # syntactic_tree.append(get_spacy_tree_1(context_token[0].tokens[1:-1]))
            syntactic_tree.append([-1])
            c_len += len(context_id)
            supporting_position.append(c_len - 1)
            contexts_id.extend(context_id)
        contexts_ids.append(contexts_id)
        syntactic_graphs.append(syntactic_tree)
        supporting_positions.append(supporting_position)
        supporting_fact_labels.append(supporting_facts_list)
        question_max_length = max(question_max_length, q_len)
        contexts_max_length = max(contexts_max_length, c_len)
        supporting_max_length = max(supporting_max_length, len(supporting_facts_list))
    dataset_length = len(question_ids)
    for i in tqdm(range(dataset_length), desc='Padding'):
        question_ids[i] += [0] * (question_max_length - len(question_ids[i]))
        contexts_ids[i] += [0] * (contexts_max_length - len(contexts_ids[i]))
        # syntactic_graphs[i] = miniGraphToBigGraph(syntactic_graphs[i], length=contexts_max_length)
        syntactic_graphs[i] = miniGraphToSmallGraph(syntactic_graphs[i], length=contexts_max_length)
        supporting_positions[i] += [0] * (supporting_max_length * 2 - len(supporting_positions[i]))
        supporting_fact_labels[i] += [-100] * (supporting_max_length - len(supporting_fact_labels[i]))
    logger.info("Created questions length max = %d.", question_max_length)
    logger.info("Created contexts length max = %d.", contexts_max_length)
    logger.info("Created supporting length max = %d.", supporting_max_length)

    if not is_torch_available():
        raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

    logger.info("Created dataset length = %d.", dataset_length)

    # Convert to Tensors and build dataset
    all_question_ids = torch.tensor(question_ids, dtype=torch.long)
    all_contexts_ids = torch.tensor(contexts_ids, dtype=torch.long)
    all_syntactic_graphs = torch.tensor(syntactic_graphs, dtype=torch.long)
    all_supporting_positions = torch.tensor(supporting_positions, dtype=torch.long)
    all_supporting_fact_labels = torch.tensor(supporting_fact_labels, dtype=torch.long)

    if mode == "train" or mode == "eval":
        dataset = TensorDataset(
            all_question_ids,
            all_contexts_ids,
            all_syntactic_graphs,
            all_supporting_positions,
            all_supporting_fact_labels,
        )
    else:
        dataset = TensorDataset(
            all_question_ids,
            all_contexts_ids,
            all_syntactic_graphs,
            all_supporting_positions,
        )

    logger.info("Saving dataset into cached file %s", temp_file)
    torch.save({"dataset": dataset, "examples": examples}, temp_file)
    return dataset, examples, tokenizer
