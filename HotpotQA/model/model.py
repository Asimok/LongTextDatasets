import logging
import math
from typing import Optional, Dict, Tuple

import torch
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel

from reloss.cls import ReLoss

from defaultModel import Attention, BertFeedForward

logger = logging.getLogger(__name__)


class SentenceChoice(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        bertModel = BertModel(config, add_pooling_layer=False)
        self.hidden_size = config.hidden_size
        self.num_layers = 3
        self.bert = bertModel.embeddings.word_embeddings
        self.encoder = bertModel.encoder
        self.encoder.layer = self.encoder.layer[:1]
        # self.q_lstm = nn.LSTM(
        #     input_size=config.hidden_size,  # 输入大小为转化后的词向量
        #     hidden_size=config.hidden_size,  # 隐藏层大小
        #     num_layers=self.num_layers,  # 堆叠层数
        #     dropout=0.5,  # 遗忘门参数
        #     bidirectional=True,  # 双向LSTM
        #     batch_first=True,
        # )
        # self.c_lstm = nn.ModuleList(nn.LSTM(
        #     input_size=config.hidden_size,  # 输入大小为转化后的词向量
        #     hidden_size=config.hidden_size,  # 隐藏层大小
        #     num_layers=1,  # 堆叠层数
        #     dropout=0.5,  # 遗忘门参数
        #     bidirectional=True,  # 双向LSTM
        #     batch_first=True,
        # ) for _ in range(self.num_layers))

        self.attention = Attention(embed_dim=config.hidden_size)
        self.liner = nn.Linear(config.hidden_size, 2, bias=True)
        # self.decoder1 = BertFeedForward(input_size=self.hidden_size,
        #                                 hidden_size=self.hidden_size,
        #                                 output_size=self.hidden_size * 2)
        # self.decoder2 = BertFeedForward(input_size=self.hidden_size * 2,
        #                                 hidden_size=self.hidden_size,
        #                                 output_size=2)  # 二分类

        self.loss_fn = ReLoss()

        # self.init_weights()

    def forward(self,
                question_id=None,
                contexts_id=None,
                syntactic_graph=None,
                supporting_position=None,
                supporting_fact_label=None,
                ):
        batch = question_id.shape[0]
        supporting_length = supporting_position.shape[1] // 2
        question_embedding = self.bert(question_id)  # [b,q,h]
        contexts_embedding = self.bert(contexts_id)  # [b,c,h]
        # LSTM
        # question_embedding, _ = self.q_lstm(question_embedding)
        # question_embedding = torch.add(*question_embedding.chunk(2, dim=-1))
        #
        # for i in range(self.num_layers):
        #     tree_embedding_forward = torch.stack(
        #         [torch.index_select(input=contexts_embedding[i], dim=0, index=syntactic_graph[i]) for i in range(batch)],
        #         dim=0)
        #     tree_embedding_backward = torch.stack(
        #         [torch.zeros_like(contexts_embedding[i]).scatter_add_(dim=0,
        #                                                               index=syntactic_graph[i].expand_as(contexts_embedding[i].T).T,
        #                                                               src=contexts_embedding[i]) for i in range(batch)],
        #         dim=0)
        #     tree_count_backward = torch.stack(
        #         [torch.zeros_like(contexts_embedding[i]).scatter_add_(dim=0,
        #                                                               index=syntactic_graph[i].expand_as(contexts_embedding[i].T).T,
        #                                                               src=torch.ones_like(contexts_embedding[i])) for i in range(batch)],
        #         dim=0)
        #     tree_count_backward = tree_count_backward.where(tree_count_backward > 0, torch.ones_like(tree_count_backward))
        #     tree_embedding_backward = tree_embedding_backward / tree_count_backward
        #
        #     new_contexts_embedding = contexts_embedding + tree_embedding_forward + tree_embedding_backward
        #     contexts_embedding, _ = self.c_lstm[i](new_contexts_embedding)
        #     contexts_embedding = torch.add(*contexts_embedding.chunk(2, dim=-1))
        # BERT
        embedding = torch.cat([question_embedding, contexts_embedding], dim=1)
        encoder = self.encoder(embedding)[0]
        contexts_attention = encoder[:, question_embedding.shape[1]:, :]  # [b,c,h]

        # contexts_attention, _ = self.attention(contexts_embedding, question_embedding)  # [b,c,h]

        # # start + end
        # supporting_attention = torch.stack(
        #     [torch.index_select(input=contexts_attention[i], dim=0, index=supporting_position[i]) for i in range(batch)],
        #     dim=0, )
        # supporting_attention = supporting_attention.reshape(batch, 2, supporting_length, -1)  # [b,2,s,h]
        # supporting_attention = torch.add(*supporting_attention.unbind(dim=1))  # [b,s,h]
        # supporting_logits = self.liner(torch.tanh(supporting_attention))  # [b,s,2]

        # [start:end]
        supporting_attention = torch.stack([
            torch.stack([
                torch.zeros_like(contexts_attention[i, 0], device=contexts_attention.device) if s == e
                else contexts_attention[i, s:e].mean(dim=0) for s, e in supporting_position[i].split(2, dim=-1)
            ], dim=0) for i in range(batch)
        ], dim=0, )

        supporting_logits = self.liner(supporting_attention)  # [b,s,2]
        # supporting_logits = self.decoder1(supporting_attention)
        # supporting_logits = self.decoder2(supporting_logits)

        if supporting_fact_label is not None:
            supporting_logits = torch.cat(supporting_logits.unbind(dim=0), dim=0)  # [b*s,2]
            supporting_fact_label = torch.cat(supporting_fact_label.unbind(dim=0), dim=-1)  # [b*s]
            # used_index = (supporting_fact_label + 100).nonzero(as_tuple=False).squeeze()
            used_index = torch.where(supporting_fact_label != -100)[0]
            # supporting_logits = supporting_logits.transpose(dim0=0, dim1=1)  # [2,b*s]
            supporting_logits = supporting_logits.index_select(dim=0, index=used_index)  # [u,2]
            supporting_fact_label = supporting_fact_label.index_select(dim=-1, index=used_index)  # [u]
            # supporting_fact_label = supporting_fact_label.where(supporting_fact_label != -100, torch.zeros_like(supporting_fact_label))
            lossF = self.loss_fn(supporting_logits, supporting_fact_label)

            loss_fct = CrossEntropyLoss(reduction='mean')
            # loss_fct = CrossEntropyLoss(reduction='sum')
            loss = loss_fct(supporting_logits, supporting_fact_label)

            # supporting_fact_label_True = supporting_fact_label.where(supporting_fact_label == 1, torch.full_like(supporting_fact_label, -100))
            # supporting_fact_label_False = supporting_fact_label.where(supporting_fact_label == 0, torch.full_like(supporting_fact_label, -100))
            # loss_True = loss_fct(supporting_logits, supporting_fact_label_True)
            # loss_False = loss_fct(supporting_logits, supporting_fact_label_False)
            # loss = (loss_True + loss_False) / 2
            # import random
            # if random.random() > 0.5:
            #     return lossF, loss,
            # return loss, lossF,
            return loss, lossF,
        supporting_logits = supporting_logits.softmax(dim=-1)
        return None, supporting_logits  # [b,s,2]


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.DEBUG
    )
    logger.debug("Debug...")
