import logging

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_bart import Attention

logger = logging.getLogger(__name__)


class SentenceChoice(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        bertModel = BertModel(config, add_pooling_layer=False)
        self.hidden_size = config.hidden_size
        self.num_layers = 3
        self.bert = bertModel.embeddings.word_embeddings
        self.q_lstm = nn.LSTM(
            input_size=config.hidden_size,  # 输入大小为转化后的词向量
            hidden_size=config.hidden_size // 2,  # 隐藏层大小
            num_layers=self.num_layers,  # 堆叠层数
            dropout=0.1,  # 遗忘门参数
            bidirectional=True,  # 双向LSTM
            batch_first=True,
        )
        self.c_lstm = nn.ModuleList(nn.LSTM(
            input_size=config.hidden_size * 2,  # 输入大小为转化后的词向量
            hidden_size=config.hidden_size // 2,  # 隐藏层大小
            num_layers=1,  # 堆叠层数
            dropout=0.1,  # 遗忘门参数
            bidirectional=True,  # 双向LSTM
            batch_first=True,
        ) for _ in range(self.num_layers))
        self.attention = Attention(embed_dim=config.hidden_size, num_heads=1)
        # self.attention = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=1, )
        self.liner = nn.Linear(config.hidden_size, 1, bias=True)

        self.init_weights()

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
        # question_embedding, _ = self.q_lstm(question_embedding)
        # # syntatic_graph = torch.stack([syntatic_graph for _ in range(self.hidden_size)], dim=2)
        # # tree_contexts_embedding = torch.gather(input=contexts_embedding, dim=1, index=syntatic_graph)
        # for i in range(self.num_layers):
        #     tree_embedding = torch.stack(
        #         [torch.index_select(input=contexts_embedding[i], dim=0, index=syntatic_graph[i]) for i in range(batch)],
        #         dim=0)
        #     new_contexts_embedding = torch.cat([contexts_embedding, tree_embedding], dim=2)
        #     contexts_embedding, _ = self.c_lstm[i](new_contexts_embedding)
        # question_embedding = question_embedding.transpose(dim0=0, dim1=1)
        # contexts_embedding = contexts_embedding.transpose(dim0=0, dim1=1)
        #
        # contexts_attention, _ = self.attention(contexts_embedding, question_embedding, )
        # # contexts_attention, _ = self.attention(contexts_embedding, question_embedding, question_embedding)
        # contexts_attention = contexts_attention.transpose(dim0=0, dim1=1)
        supporting_attention = torch.stack(
            [torch.index_select(input=contexts_embedding[i], dim=0, index=supporting_position[i]) for i in range(batch)],
            dim=0, )
        supporting_attention = supporting_attention.reshape(batch, 2, supporting_length, -1)
        supporting_logits = self.liner(supporting_attention).squeeze(dim=-1)
        if supporting_fact_label is not None:
            loss_fct = CrossEntropyLoss()
            # loss = loss_fct(supporting_logits, supporting_fact_label)
            supporting_fact_label_True = supporting_fact_label.where(supporting_fact_label == 1, torch.full_like(supporting_fact_label, -100))
            supporting_fact_label_False = supporting_fact_label.where(supporting_fact_label == 0, torch.full_like(supporting_fact_label, -100))
            # supporting_logits_softmax = supporting_logits.softmax(dim=1)
            # supporting_logits_False = supporting_logits_softmax[:, 0, :]
            # supporting_logits_True = supporting_logits_softmax[:, 1, :]
            # torch.ones_like(supporting_logits).where(supporting_fact_label != 0, supporting_logits_False)
            # torch.ones_like(supporting_logits).where(supporting_fact_label != 1, supporting_logits_True)
            loss_True = loss_fct(supporting_logits, supporting_fact_label_True)
            loss_False = loss_fct(supporting_logits, supporting_fact_label_False)
            loss = (loss_True + loss_False) / 2
            return loss, None
        return None, supporting_logits
