import copy
import math
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from reloss.cls import ReLoss
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.modules.loss import _WeightedLoss
from transformers import BertPreTrainedModel, BertModel

from LongTextModels.config import config
from LongTextModels.tools.logger import get_logger

logger = get_logger(log_name="model")


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


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

        # 注释
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.bmm(attn_weights, v)
        assert attn_output.size() == (bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        # attn_output = self.out_proj(attn_weights)

        if output_attentions:
            return attn_output, attn_weights
        return attn_output, None


class BertFeedForward(nn.Module):
    def __init__(self, _config, input_size, intermediate_size, output_size):
        super(BertFeedForward, self).__init__()
        self.dense = nn.Linear(input_size, intermediate_size)
        self.affine = nn.Linear(intermediate_size, output_size)
        self.act_fn = gelu
        self.LayerNorm = BERTLayerNorm(_config)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.affine(hidden_states)
        return hidden_states


class BERTLayerNorm(nn.Module):
    def __init__(self, _config, variance_epsilon=1e-12):
        """
        Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BERTLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(_config.hidden_size))
        self.beta = nn.Parameter(torch.zeros(_config.hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class WeightedFocalLoss(nn.Module):
    """Non weighted version of Focal Loss"""

    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha, 1 - alpha]))
        # self.alpha = torch.tensor([alpha, 1 - alpha]).to('cuda')
        self.gamma = gamma

    def forward(self, inputs, targets):
        # 应该去掉不参与训练的-100 数据部分
        input_softmax = F.softmax(inputs, dim=1)
        input_softmax = input_softmax[:, 1]
        input_softmax = input_softmax.type(torch.float)
        targets = targets.type(torch.float)
        BCE_loss = F.binary_cross_entropy_with_logits(input=input_softmax, target=targets, reduction='sum')

        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


class CrossEntropyLossForFocalLoss(_WeightedLoss):
    """
    Examples::
        # >>> loss = nn.CrossEntropyLoss()
        # >>> input = torch.randn(3, 5, requires_grad=True)
        # >>> target = torch.empty(3, dtype=torch.long).random_(5)
        # >>> output = loss(input, target)
        # >>> output.backward()
    """
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', alpha=.4, gamma=2) -> None:
        super(CrossEntropyLossForFocalLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.alpha = nn.Parameter(torch.tensor([alpha, 1 - alpha]), requires_grad=False)
        self.gamma = gamma

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert self.weight is None or isinstance(self.weight, Tensor)
        ce_loss = F.cross_entropy(input, target, weight=self.weight,
                                  ignore_index=self.ignore_index, reduction=self.reduction)
        targets = target.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-ce_loss)
        F_loss = at * (1 - pt) ** self.gamma * ce_loss
        return F_loss.mean()


class SentenceChoice(BertPreTrainedModel):
    def __init__(self, _config):
        super().__init__(_config)
        bertModel = BertModel(_config, add_pooling_layer=False)
        self.hidden_size = _config.hidden_size
        self.num_layers = 3
        self.bert = bertModel.embeddings.word_embeddings
        self.RELoss_fn = ReLoss(pretrained=False)
        # self.focal_loss = WeightedFocalLoss()
        self.focal_loss = CrossEntropyLossForFocalLoss()
        self.q_lstm = nn.LSTM(
            input_size=self.hidden_size,  # 输入大小为转化后的词向量
            hidden_size=self.hidden_size // 2,  # 隐藏层大小 双向 输出维度x2
            num_layers=1,  # 堆叠层数
            dropout=0.1,  # 遗忘门参数
            bidirectional=True,  # 双向LSTM
            batch_first=True,
        )

        self.c_lstm = nn.LSTM(
            input_size=_config.hidden_size,  # 输入大小为转化后的词向量
            hidden_size=_config.hidden_size // 2,  # 隐藏层大小
            num_layers=1,  # 堆叠层数
            dropout=0.2,  # 遗忘门参数
            bidirectional=True,  # 双向LSTM
            batch_first=True,
        )

        # Attention
        self.attention = Attention(embed_dim=_config.hidden_size)

        self.decoder1 = BertFeedForward(_config, input_size=self.hidden_size * 1,
                                        intermediate_size=self.hidden_size * 1, output_size=self.hidden_size)
        self.decoder2 = BertFeedForward(_config, input_size=self.hidden_size,
                                        intermediate_size=self.hidden_size, output_size=2)  # 二分类

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

        # 单层句法分析树
        tree_embedding_forward = torch.stack(
            [torch.index_select(input=contexts_embedding[i], dim=0, index=syntactic_graph[i]) for i in
             range(batch)],
            dim=0)
        tree_embedding_backward = torch.stack(
            [torch.zeros_like(contexts_embedding[i]).scatter_add_(dim=0,
                                                                  index=syntactic_graph[i].expand_as(
                                                                      contexts_embedding[i].T).T,
                                                                  src=contexts_embedding[i]) for i in range(batch)],
            dim=0)
        tree_count_backward = torch.stack(
            [torch.zeros_like(contexts_embedding[i]).scatter_add_(dim=0,
                                                                  index=syntactic_graph[i].expand_as(
                                                                      contexts_embedding[i].T).T,
                                                                  src=torch.ones_like(contexts_embedding[i])) for i
             in range(batch)],
            dim=0)
        tree_count_backward = tree_count_backward.where(tree_count_backward > 0,
                                                        torch.ones_like(tree_count_backward))
        tree_embedding_backward = tree_embedding_backward / tree_count_backward

        new_contexts_embedding = contexts_embedding + tree_embedding_forward + tree_embedding_backward
        contexts_embedding, _ = self.c_lstm(new_contexts_embedding)

        # Attention
        contexts_attention, _ = self.attention(contexts_embedding, question_embedding)  # [b,c,h]
        # 线性层

        supporting_logits = self.decoder1(contexts_attention)
        supporting_logits = self.decoder2(supporting_logits)

        """
        两个下标[s,e]标志句子的起始和结束位置
        这里采用平均池化 获得句子的表示
        """
        # 单独取出句子头尾
        start_signal = torch.tensor([(2 * i) for i in range(supporting_length)]).to(config.device)
        end_signal = torch.tensor([(2 * i) + 1 for i in range(supporting_length)]).to(config.device)
        start_sentence_supporting_position = torch.stack(
            [torch.index_select(input=supporting_position[i], dim=0, index=start_signal) for i in
             range(batch)],
            dim=0)
        end_sentence_supporting_position = torch.stack(
            [torch.index_select(input=supporting_position[i], dim=0, index=end_signal) for i in
             range(batch)],
            dim=0)
        # 切片取出整个句子的tensor 再做平均池化
        sentences = []
        for i in range(batch):
            supporting_logits_batch = supporting_logits[i]
            support = []
            for j in range(len(start_sentence_supporting_position[i])):
                s = start_sentence_supporting_position[i][j]
                e = end_sentence_supporting_position[i][j]
                support.append(torch.mean(supporting_logits_batch[s:e, :], dim=0))
            support_tensor = torch.stack(support, dim=0)
            sentences.append(support_tensor)
        supporting_logits_sentence = torch.stack(sentences, dim=0)

        # 转换为  batch  x class x seq
        supporting_logits_for_ce_loss = supporting_logits_sentence.permute(0, 2, 1)
        supporting_logits_for_focal_loss = supporting_logits_sentence.permute(0, 2, 1)
        supporting_fact_label_for_ce_loss = copy.deepcopy(supporting_fact_label)
        if supporting_fact_label is not None:
            # 对于序列标注来说，需要reshape一下
            # supporting_logits = supporting_logits.reshape(-1, 2)  # 两个类别
            # supporting_fact_label = supporting_fact_label.view(-1)
            """
            加上re_loss
            """
            supporting_logits = torch.cat(supporting_logits_sentence.unbind(dim=0), dim=0)  # [b*s,2]
            supporting_fact_label = torch.cat(supporting_fact_label.unbind(dim=0), dim=-1)  # [b*s]

            used_index = torch.where(supporting_fact_label != -100)[0]  # 参与训练的句子

            supporting_logits = supporting_logits.index_select(dim=0, index=used_index)  # [u,2]
            supporting_fact_label = supporting_fact_label.index_select(dim=-1, index=used_index)  # [u]

            re_loss = self.RELoss_fn(supporting_logits, supporting_fact_label)
            """
            CrossEntropyLoss
            """
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(supporting_logits_for_ce_loss, supporting_fact_label_for_ce_loss)
            """
            focal loss
            魔改自官方交叉熵损失
            """
            focal_loss = self.focal_loss(supporting_logits, supporting_fact_label)
            # print(focal_loss)
            return re_loss, loss, focal_loss, supporting_logits_for_ce_loss

        return None, None, None, supporting_logits_for_ce_loss
