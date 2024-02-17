from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel


class BertForExtractiveQA(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(self, batch_inputs, start_positions=None, end_positions=None):
        bert_output = self.bert(**batch_inputs)  # (batch_size, seq_len, hidden_size)
        sequence_output = (
            bert_output.last_hidden_state
        )  # (batch_size, seq_len, hidden_size)
        sequence_output = self.dropout(
            sequence_output
        )  # (batch_size, seq_len, hidden_size)
        # 获得每个 token 为答案起始、结束位置的分数
        logits = self.classifier(
            sequence_output
        )  # (batch_size, seq_len, num_labels)  num_labels=2

        start_logits, end_logits = logits.split(
            1, dim=-1
        )  # (batch_size, seq_len, 1)，(batch_size, seq_len, 1)
        # contiguous()：重新分配内存，确保张量是连续的
        start_logits = start_logits.squeeze(-1).contiguous()  # (batch_size, seq_len)
        end_logits = end_logits.squeeze(-1).contiguous()  # (batch_size, seq_len)

        loss = None
        # 在整个序列所有的 L 个 token 上选出一个 token 作为答案的起始 / 结束，相当于是在进行一个 L 分类问题
        # 所以可以分别在起始和结束的输出上运用交叉熵来计算损失，然后取两个损失的平均值作为模型的整体损失
        if start_positions is not None and end_positions is not None:
            loss_fct = CrossEntropyLoss()
            start_loss = loss_fct(
                start_logits, start_positions
            )  # start_positions：(batch_size, )
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2

        return loss, start_logits, end_logits
