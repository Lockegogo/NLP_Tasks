from torch import nn
from transformers import (
    BertModel,
    BertPreTrainedModel,  # 抽象基类，继承它可以自动获得一些方法，例如 from_pretrained
    RobertaModel,
    RobertaPreTrainedModel,
)

# 继承  Transformers 库中的预训练模型来创建自己的模型
# 好处：可以更灵活地操作模型细节，例如这里的 Dropout 层就可以直接加载 BERT 模型自带的参数值
class BertForPairwiseCLS(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)
        self.post_init()

    def forward(self, batch_inputs, labels=None):
        outputs = self.bert(**batch_inputs)
        cls_vectors = outputs.last_hidden_state[:, 0, :]
        cls_vectors = self.dropout(cls_vectors)
        logits = self.classifier(cls_vectors)
        # 与 transformers 库类似，我们将模型损失的计算也包含进模型本身
        # 这样在训练循环中我们就可以直接使用模型返回的损失进行反向传播
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return loss, logits


class RobertaForPairwiseCLS(RobertaPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)
        self.post_init()

    def forward(self, batch_inputs, labels=None):
        outputs = self.roberta(**batch_inputs)
        cls_vectors = outputs.last_hidden_state[:, 0, :]
        cls_vectors = self.dropout(cls_vectors)
        logits = self.classifier(cls_vectors)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return loss, logits
