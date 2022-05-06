import logging
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from transformers import BertPreTrainedModel, BertModel
from transformers.models.bart.modeling_bart import BartAttention

from LongTextModels.tools.logger import get_logger

logger = get_logger(log_name="model")


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertFeedForward(nn.Module):
    def __init__(self, config, input_size, intermediate_size, output_size):
        super(BertFeedForward, self).__init__()
        self.dense = nn.Linear(input_size, intermediate_size)
        self.affine = nn.Linear(intermediate_size, output_size)
        self.act_fn = gelu
        # torch.nn.functional.relu
        self.LayerNorm = BERTLayerNorm(config)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.affine(hidden_states)
        return hidden_states


class BERTLayerNorm(nn.Module):
    def __init__(self, config, variance_epsilon=1e-12):
        """
        Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BERTLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(config.hidden_size))
        self.beta = nn.Parameter(torch.zeros(config.hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class SentenceChoice(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        bertModel = BertModel(config, add_pooling_layer=False)

        self.hidden_size = config.hidden_size
        self.num_layers = 3
        self.bert = bertModel.embeddings.word_embeddings
        self.q_lstm = nn.LSTM(
            input_size=self.hidden_size,  # 输入大小为转化后的词向量
            hidden_size=self.hidden_size // 2,  # 隐藏层大小 双向 输出维度x2
            num_layers=self.num_layers,  # 堆叠层数
            dropout=0.5,  # 遗忘门参数
            bidirectional=True,  # 双向LSTM
            batch_first=True,
        )
        self.c_lstm = nn.ModuleList(nn.LSTM(
            input_size=config.hidden_size * 2,  # 输入大小为转化后的词向量
            hidden_size=config.hidden_size // 2,  # 隐藏层大小
            num_layers=1,  # 堆叠层数
            dropout=0.5,  # 遗忘门参数
            bidirectional=True,  # 双向LSTM
            batch_first=True,
        ) for _ in range(self.num_layers))

        # Attention
        # weight_w即为公式中的h_s(参考系)
        # nn. Parameter的作用是参数是需要梯度的
        self.weight_W = nn.Parameter(torch.Tensor(self.hidden_size, 2 * self.hidden_size))
        self.weight_proj = nn.Parameter(torch.Tensor(2 * self.hidden_size, 1))
        # 对weight_W、weight_proj进行初始化
        nn.init.uniform_(self.weight_W, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj, -0.1, 0.1)
        # 双向 lstm 维度x2
        self.attention = BartAttention(embed_dim=self.hidden_size * 2, num_heads=1)

        # self.decoder1 = nn.Linear(self.hidden_size * 1, self.hidden_size)
        # self.decoder2 = nn.Linear(self.hidden_size, 1)

        self.decoder1 = BertFeedForward(config, input_size=self.hidden_size * 1,
                                        intermediate_size=self.hidden_size, output_size=self.hidden_size)
        self.decoder2 = BertFeedForward(config, input_size=self.hidden_size,
                                        intermediate_size=self.hidden_size, output_size=1)

        self.init_weights()

    def forward(self,
                question_id=None,
                contexts_id=None,
                syntactic_graph=None,
                supporting_position=None,
                supporting_fact_label=None,
                ):
        batch = question_id.shape[0]
        supporting_length = supporting_position.shape[1] // 2
        question_embedding = self.bert(question_id)
        contexts_embedding = self.bert(contexts_id)
        # LSTM
        question_embedding, _ = self.q_lstm(question_embedding)

        for i in range(self.num_layers):
            tree_embedding = torch.stack(
                [torch.index_select(input=contexts_embedding[i], dim=0, index=syntactic_graph[i]) for i in
                 range(batch)],
                dim=0)
            new_contexts_embedding = torch.cat([contexts_embedding, tree_embedding], dim=2)
            contexts_embedding, _ = self.c_lstm[i](new_contexts_embedding)

        # TODO 新加
        # Attention
        states = torch.cat([contexts_embedding, question_embedding], dim=1)
        # states与self.weight_W矩阵相乘，然后做tanh
        u = torch.tanh(torch.matmul(states, self.weight_W))
        # u与self.weight_proj矩阵相乘,得到score
        att = torch.matmul(u, self.weight_proj)
        # softmax
        att_score = F.softmax(att, dim=1)
        # 加权求和
        scored_x = states * att_score
        # encoding = torch.sum(scored_x, dim=1)
        # 线性层

        supporting_logits = self.decoder1(scored_x)
        supporting_logits = self.decoder2(supporting_logits).squeeze(dim=-1)
        supporting_attention = torch.stack(
            [torch.index_select(input=supporting_logits[i], dim=0, index=supporting_position[i]) for i in
             range(batch)],
            dim=0)
        supporting_logits = supporting_attention.reshape(batch, 2, supporting_length, -1).squeeze(
            dim=-1)  # batch  x class x seq

        if supporting_fact_label is not None:
            # 对于序列标注来说，需要reshape一下
            supporting_logits = supporting_logits.reshape(-1, 2)  # 两个类别
            supporting_fact_label = supporting_fact_label.view(-1)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(supporting_logits, supporting_fact_label)

            # supporting_fact_label_True = supporting_fact_label.where(supporting_fact_label == 1,
            #                                                          torch.full_like(supporting_fact_label, -100))
            # supporting_fact_label_False = supporting_fact_label.where(supporting_fact_label == 0,
            #                                                           torch.full_like(supporting_fact_label, -100))
            # loss_True = loss_fct(supporting_logits, supporting_fact_label_True)
            # loss_False = loss_fct(supporting_logits, supporting_fact_label_False)
            # loss = 0.5 * loss_True + 0.5 * loss_False
            return loss, None

        return None, supporting_logits
