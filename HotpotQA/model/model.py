import logging
import math
from typing import Optional, Dict, Tuple

import torch
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel

# from transformers.modeling_bart import Attention

logger = logging.getLogger(__name__)


class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, embed_dim, bias=True, ):
        super().__init__()
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
            self,
            query,
            key: Tensor,
            output_attentions=False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time(SeqLen) x Batch x Channel"""
        bsz, tgt_len, embed_dim = query.size()

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(key)

        src_len = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert attn_weights.size() == (bsz, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.bmm(attn_weights, v)
        assert attn_output.size() == (bsz, tgt_len, embed_dim)
        # attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        # attn_output = self.out_proj(attn_weights)

        if output_attentions:
            return attn_output, attn_weights
        return attn_output, None


class SentenceChoice(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        bertModel = BertModel(config, add_pooling_layer=False)
        self.hidden_size = config.hidden_size
        self.num_layers = 3
        self.bert = bertModel.embeddings.word_embeddings
        self.q_lstm = nn.LSTM(
            input_size=config.hidden_size,  # 输入大小为转化后的词向量
            hidden_size=config.hidden_size,  # 隐藏层大小
            num_layers=self.num_layers,  # 堆叠层数
            dropout=0.8,  # 遗忘门参数
            bidirectional=True,  # 双向LSTM
            batch_first=True,
        )
        self.c_lstm = nn.ModuleList(nn.LSTM(
            input_size=config.hidden_size,  # 输入大小为转化后的词向量
            hidden_size=config.hidden_size,  # 隐藏层大小
            num_layers=1,  # 堆叠层数
            dropout=0.8,  # 遗忘门参数
            bidirectional=True,  # 双向LSTM
            batch_first=True,
        ) for _ in range(self.num_layers))
        self.attention = Attention(embed_dim=config.hidden_size)
        # self.attention = Attention(embed_dim=config.hidden_size,num_heads=1)
        # self.attention = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=1, )
        self.liner = nn.Linear(config.hidden_size, 1, bias=True)

        # self.init_weights()

    def forward(self,
                question_id=None,
                contexts_id=None,
                syntatic_graph=None,
                supporting_position=None,
                supporting_fact_label=None,
                ):
        batch = question_id.shape[0]
        supporting_length = supporting_position.shape[1] // 2
        question_embedding = self.bert(question_id)
        contexts_embedding = self.bert(contexts_id)
        # LSTM
        question_embedding, _ = self.q_lstm(question_embedding)
        question_embedding = torch.add(*question_embedding.chunk(2, dim=-1))
        # syntatic_graph = torch.stack([syntatic_graph for _ in range(self.hidden_size)], dim=2)
        # tree_contexts_embedding = torch.gather(input=contexts_embedding, dim=1, index=syntatic_graph)
        for i in range(self.num_layers):
            tree_embedding_forward = torch.stack(
                [torch.index_select(input=contexts_embedding[i], dim=0, index=syntatic_graph[i]) for i in range(batch)],
                dim=0)
            tree_embedding_backward = torch.stack(
                [torch.zeros_like(contexts_embedding[i]).scatter_add_(dim=0,
                                                                      index=syntatic_graph[i].expand_as(contexts_embedding[i].T).T,
                                                                      src=contexts_embedding[i]) for i in range(batch)],
                dim=0)
            tree_count_backward = torch.stack(
                [torch.zeros_like(contexts_embedding[i]).scatter_add_(dim=0,
                                                                      index=syntatic_graph[i].expand_as(contexts_embedding[i].T).T,
                                                                      src=torch.ones_like(contexts_embedding[i])) for i in range(batch)],
                dim=0)
            tree_count_backward = tree_count_backward.where(tree_count_backward > 0, torch.ones_like(tree_count_backward))
            tree_embedding_backward = tree_embedding_backward / tree_count_backward

            new_contexts_embedding = contexts_embedding + tree_embedding_forward + tree_embedding_backward
            contexts_embedding, _ = self.c_lstm[i](new_contexts_embedding)
            contexts_embedding = torch.add(*contexts_embedding.chunk(2, dim=-1))
        contexts_attention, _ = self.attention(contexts_embedding, question_embedding)

        # question_embedding = question_embedding.transpose(dim0=0, dim1=1)
        # contexts_embedding = contexts_embedding.transpose(dim0=0, dim1=1)
        # contexts_attention, _ = self.attention(contexts_embedding, question_embedding)
        # # _, contexts_attention = self.attention(contexts_embedding, question_embedding, output_attentions=True)
        # contexts_attention=contexts_attention.squeeze()
        # contexts_attention = contexts_attention.transpose(dim0=0, dim1=1)

        supporting_attention = torch.stack(
            [torch.index_select(input=contexts_attention[i], dim=0, index=supporting_position[i]) for i in range(batch)],
            dim=0, )
        supporting_attention = supporting_attention.reshape(batch, 2, supporting_length, -1)
        supporting_logits = self.liner(supporting_attention).squeeze(dim=-1)
        if supporting_fact_label is not None:
            loss_fct = CrossEntropyLoss(reduction='mean')
            # loss_fct = CrossEntropyLoss(reduction='sum')
            loss = loss_fct(supporting_logits, supporting_fact_label)
            # loss1 = loss_fct(supporting_logits, supporting_fact_label)
            #
            # print('\n', loss.item() / loss1.item())

            # supporting_fact_label_True = supporting_fact_label.where(supporting_fact_label == 1, torch.full_like(supporting_fact_label, -100))
            # supporting_fact_label_False = supporting_fact_label.where(supporting_fact_label == 0, torch.full_like(supporting_fact_label, -100))
            # # supporting_logits_softmax = supporting_logits.softmax(dim=1)
            # # supporting_logits_False = supporting_logits_softmax[:, 0, :]
            # # supporting_logits_True = supporting_logits_softmax[:, 1, :]
            # # torch.ones_like(supporting_logits).where(supporting_fact_label != 0, supporting_logits_False)
            # # torch.ones_like(supporting_logits).where(supporting_fact_label != 1, supporting_logits_True)
            # loss_True = loss_fct(supporting_logits, supporting_fact_label_True)
            # loss_False = loss_fct(supporting_logits, supporting_fact_label_False)
            # # print(loss_False.item(), loss_True.item())
            # loss = (loss_True + loss_False) / 2
            # # loss = (loss_True * 1.3 + loss_False * 0.7
            return loss, None
        return None, supporting_logits


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.DEBUG
    )
    logger.debug("Debug...")

# def gelu(x):
#     """Implementation of the gelu activation function.
#         For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
#         0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
#     """
#     return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
#
#
# class BertFeedForward(nn.Module):
#     def __init__(self, config, input_size, intermediate_size, output_size):
#         super(BertFeedForward, self).__init__()
#         self.dense = nn.Linear(input_size, intermediate_size)
#         self.affine = nn.Linear(intermediate_size, output_size)
#         self.act_fn = gelu
#         # torch.nn.functional.relu
#         self.LayerNorm = BERTLayerNorm(config)
#
#     def forward(self, hidden_states):
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.act_fn(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states)
#         hidden_states = self.affine(hidden_states)
#         return hidden_states
#
#
# class BERTLayerNorm(nn.Module):
#     def __init__(self, config, variance_epsilon=1e-12):
#         """
#         Construct a layernorm module in the TF style (epsilon inside the square root).
#         """
#         super(BERTLayerNorm, self).__init__()
#         self.gamma = nn.Parameter(torch.ones(config.hidden_size))
#         self.beta = nn.Parameter(torch.zeros(config.hidden_size))
#         self.variance_epsilon = variance_epsilon
#
#     def forward(self, x):
#         u = x.mean(-1, keepdim=True)
#         s = (x - u).pow(2).mean(-1, keepdim=True)
#         x = (x - u) / torch.sqrt(s + self.variance_epsilon)
#         return self.gamma * x + self.beta
#
#
# class SentenceChoice(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         bertModel = BertModel(config, add_pooling_layer=False)
#
#         self.hidden_size = config.hidden_size
#         self.num_layers = 3
#         self.bert = bertModel.embeddings.word_embeddings
#         self.q_lstm = nn.LSTM(
#             input_size=self.hidden_size,  # 输入大小为转化后的词向量
#             hidden_size=self.hidden_size,  # 隐藏层大小 双向 输出维度x2
#             num_layers=self.num_layers,  # 堆叠层数
#             dropout=0.5,  # 遗忘门参数
#             bidirectional=True,  # 双向LSTM
#             batch_first=True,
#         )
#         self.c_lstm = nn.LSTM(
#             input_size=self.hidden_size,  # 输入大小为转化后的词向量
#             hidden_size=self.hidden_size,  # 隐藏层大小 双向 输出维度x2
#             num_layers=self.num_layers,  # 堆叠层数
#             dropout=0.5,  # 遗忘门参数
#             bidirectional=True,  # 双向LSTM
#             batch_first=True,
#         )
#         # self.c_lstm = nn.ModuleList(nn.LSTM(
#         #     input_size=config.hidden_size * 2,  # 输入大小为转化后的词向量
#         #     hidden_size=config.hidden_size,  # 隐藏层大小
#         #     num_layers=1,  # 堆叠层数
#         #     dropout=0.5,  # 遗忘门参数
#         #     bidirectional=True,  # 双向LSTM
#         #     batch_first=True,
#         # ) for _ in range(self.num_layers))
#         # 双向 lstm 维度x2
#         self.attention = Attention(embed_dim=self.hidden_size * 2, num_heads=1)
#         # self.liner = nn.Linear(in_features=self.hidden_size, out_features=1, bias=True)
#         # self.liner = BertFeedForward(config, input_size=self.hidden_size,
#         #                              intermediate_size=self.hidden_size, output_size=1)
#
#         # self.decoder1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
#         # self.decoder2 = nn.Linear(self.hidden_size, 1)
#
#         self.decoder1 = BertFeedForward(config, input_size=self.hidden_size * 2,
#                                         intermediate_size=self.hidden_size, output_size=self.hidden_size)
#         self.decoder2 = BertFeedForward(config, input_size=self.hidden_size,
#                                         intermediate_size=self.hidden_size, output_size=1)
#
#         # self.init_weights()
#
#     def forward(self,
#                 question_id=None,
#                 contexts_id=None,
#                 syntatic_graph=None,
#                 supporting_position=None,
#                 supporting_fact_label=None,
#                 ):
#         batch = question_id.shape[0]
#         supporting_length = supporting_position.shape[1] // 2
#         question_embedding = self.bert(question_id)
#         contexts_embedding = self.bert(contexts_id)
#         # LSTM
#         question_embedding, _ = self.q_lstm(question_embedding)
#         contexts_embedding, _ = self.c_lstm(contexts_embedding)
#         # syntatic_graph = torch.stack([syntatic_graph for _ in range(self.hidden_size)], dim=2)
#         # tree_contexts_embedding = torch.gather(input=contexts_embedding, dim=1, index=syntatic_graph)
#         # for i in range(self.num_layers):
#         #     tree_embedding = torch.stack(
#         #         [torch.index_select(input=contexts_embedding[i], dim=0, index=syntatic_graph[i]) for i in
#         #          range(batch)],
#         #         dim=0)
#         #     new_contexts_embedding = torch.cat([contexts_embedding, tree_embedding], dim=2)
#         #     contexts_embedding, _ = self.c_lstm[i](new_contexts_embedding)
#         # question_embedding = question_embedding.transpose(dim0=0, dim1=1)
#         # contexts_embedding = contexts_embedding.transpose(dim0=0, dim1=1)
#
#         # contexts_attention = self.attention(contexts_embedding, question_embedding)
#         # # contexts_attention, _ = self.attention(contexts_embedding, question_embedding, question_embedding)
#         # # contexts_attention = contexts_attention[0].transpose(dim0=0, dim1=1)
#         # contexts_attention = contexts_attention[0]
#         # supporting_attention = torch.stack(
#         #     [torch.index_select(input=contexts_attention[i], dim=0, index=supporting_position[i]) for i in
#         #      range(batch)],
#         #     dim=0)
#         # supporting_attention = supporting_attention.reshape(batch, 2, supporting_length, -1)
#         # # supporting_attention = supporting_attention.reshape(batch, supporting_length, -1)
#
#         # TODO 新加
#
#         supporting_logits = self.decoder1(contexts_embedding)
#         supporting_logits = self.decoder2(supporting_logits).squeeze(dim=-1)
#         supporting_attention = torch.stack(
#             [torch.index_select(input=supporting_logits[i], dim=0, index=supporting_position[i]) for i in
#              range(batch)],
#             dim=0)
#         supporting_logits = supporting_attention.reshape(batch, 2, supporting_length, -1).squeeze(
#             dim=-1)  # batch  x class x seq
#         # 交换维度
#         # supporting_logits = supporting_attention.transpose(dim0=1, dim1=2)
#         # supporting_logits = self.liner(supporting_attention)
#         # supporting_fact_label = torch.where(supporting_fact_label == -100, 0, supporting_fact_label)  # batch x seq
#
#
#         if supporting_fact_label is not None:
#             # 对于序列标注来说，需要reshape一下
#             supporting_logits = supporting_logits.reshape(-1, 2)  # 两个类别
#             supporting_fact_label = supporting_fact_label.view(-1)
#             loss_fct = CrossEntropyLoss()
#             # supporting_fact_label_True = supporting_fact_label.where(supporting_fact_label != 1,
#             #                                                          torch.full_like(supporting_fact_label, -100))
#             # supporting_fact_label_False = supporting_fact_label.where(supporting_fact_label != 0,
#             #                                                           torch.full_like(supporting_fact_label, -100))
#             # loss_True = loss_fct(supporting_logits, supporting_fact_label_True)
#             # loss_False = loss_fct(supporting_logits, supporting_fact_label_False)
#             # loss = loss_True + loss_False
#             loss = loss_fct(supporting_logits, supporting_fact_label)
#             return loss, None
#
#         return None, supporting_logits
