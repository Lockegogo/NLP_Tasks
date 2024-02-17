import sys

sys.path.append("../")
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel
from tools import FullyConnectedLayer, CRF


class BertForNER(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.use_ffnn_layer = args.use_ffnn_layer
        if self.use_ffnn_layer:
            self.ffnn_size = (
                args.ffnn_size if args.ffnn_size != -1 else config.hidden_size
            )
            self.mlp = FullyConnectedLayer(
                config, config.hidden_size, self.ffnn_size, config.hidden_dropout_prob
            )
        else:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(
            self.ffnn_size if args.use_ffnn_layer else config.hidden_size,
            self.num_labels,
        )
        self.post_init()

    def forward(self, batch_inputs, labels=None):
        bert_output = self.bert(
            **batch_inputs
        )  # shape：(batch_size, seq_len, hidden_size)
        sequence_output = bert_output.last_hidden_state
        if self.use_ffnn_layer:
            sequence_output = self.mlp(sequence_output)
        else:
            sequence_output = self.dropout(sequence_output)
        logits = self.classifier(
            sequence_output
        )  # shape：(batch_size, seq_len, num_labels)

        loss = None
        if labels is not None:  # shape：(batch_size, seq_len)
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            # 获取 batch 中的 attention mask，用于计算损失时排除填充位置
            attention_mask = batch_inputs.get(
                "attention_mask"
            )  # shape: (batch_size, seq_len)
            if attention_mask is not None:
                # 将 attention mask 展平成一维张量，然后创建布尔掩码，1 表示是真实 token (对应 True)，0 表示是填充 token (对应 False)
                active_loss = (
                    attention_mask.view(-1) == 1
                )  # shape: (batch_size * seq_len)
                # 将模型的输出 logits 展平为二维张量，并根据之前创建的布尔掩码选取有效的部分
                active_logits = logits.view(-1, self.num_labels)[
                    active_loss
                ]  # shape: (num_active_tokens, num_labels)
                active_labels = labels.view(-1)[
                    active_loss
                ]  # shape: (num_active_tokens,)
                # 直接对展平的向量计算损失，不需要进行维度变换
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits


class BertCrfForNER(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.use_ffnn_layer = args.use_ffnn_layer
        if self.use_ffnn_layer:
            self.ffnn_size = (
                args.ffnn_size if args.ffnn_size != -1 else config.hidden_size
            )
            self.mlp = FullyConnectedLayer(
                config, config.hidden_size, self.ffnn_size, config.hidden_dropout_prob
            )
        else:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(
            self.ffnn_size if args.use_ffnn_layer else config.hidden_size,
            self.num_labels,
        )
        self.crf = CRF(num_tags=self.num_labels, batch_first=True)
        self.post_init()

    def forward(self, batch_inputs, labels=None):
        bert_output = self.bert(**batch_inputs)
        sequence_output = bert_output.last_hidden_state
        if self.use_ffnn_layer:
            sequence_output = self.mlp(sequence_output)
        else:
            sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = -1 * self.crf(
                emissions=logits, tags=labels, mask=batch_inputs.get("attention_mask")
            )
        return loss, logits
