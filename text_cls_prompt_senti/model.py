import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel
from transformers.activations import ACT2FN

# sequence_output, 1, batch_mask_idxs.unsqueeze(-1)
def batched_index_select(input, dim, index):
    """
    批量索引选择：在指定维度 dim 上，根据 index 中的索引值，从 input 中抽取对应的元素

    input: [batch_size, seq_len, hidden_size]
    dim: 1
    index: [batch_size, 1]
    """
    for i in range(1, len(input.shape)):
        if i != dim:
            index = index.unsqueeze(i)  # index shape： [batch_size, 1, 1]
    expanse = list(input.shape)  # expanse = [batch_size, seq_len, hidden_size]， 列表中三个元素，分别对应三个维度的大小
    expanse[0] = -1
    expanse[dim] = -1  # expanse = [-1, -1, hidden_size]
    index = index.expand(expanse)  # 在指定维度上扩展，index shape： [batch_size, 1, hidden_size]
    return torch.gather(input, dim, index)  # 在指定维度 dim 上，根据 index 中的索引值，从 input 中抽取对应的元素


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)  # 线性变换
        hidden_states = self.transform_act_fn(hidden_states)  # 激活函数
        hidden_states = self.LayerNorm(hidden_states)  # LayerNorm
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)  # 对 BertPredictionHeadTransform 的输出进行线性变换，以预测每个词在词汇表中的概率
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertForPrompt(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        # Initialize weights and apply final processing
        self.post_init()

    # 下面这两个函数负责调整模型的 MLM head
    def get_output_embeddings(self):
        return self.cls.predictions.decoder
    # 会在模型实例化时自动调用吗？
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(self, batch_inputs, batch_mask_idxs, label_word_id, labels=None):
        bert_output = self.bert(**batch_inputs)
        sequence_output = (
            bert_output.last_hidden_state
        )  # [batch_size, seq_len, hidden_size]
        # 从 BERT 的输出序列中抽取出 [MASK] token 对应的表示
        batch_mask_reps = batched_index_select(
            sequence_output, 1, batch_mask_idxs.unsqueeze(-1)
        ).squeeze(1)  # [batch_size, 1, hidden_size] 如果 [MASK] 只有一个的话
        # 然后使用 MLM head 预测出该 [MASK] token 对应词表中每个 token 的分数
        # 但是我们只返回类别对应 label words 的分数用于分类
        pred_scores = self.cls(batch_mask_reps)[:, label_word_id]  # [batch_size, vocab_size] -> [batch_size, 2]

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(pred_scores, labels)
        return loss, pred_scores
