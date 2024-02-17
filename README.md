

# README

## 文本分类

**文本分类（Text Classification）**： `[CLS]+ 句子 +[SEP]` 传入到 BERT 中，然后取出 [CLS] 的 hidden 向量做分类就行。

<img src="https://img2023.cnblogs.com/blog/1641623/202402/1641623-20240217104725384-1051388670.png" alt="4b31fb6f-13ec-4764-aee1-7a8db5afdab9" width="50%" height="50%">

## 文本匹配

**文本匹配（Text Matching）**：句子对的分类任务，比如相似匹配是判断两个句子是否相似；自然语言推理是判断两个句子之间的逻辑关系（蕴含、中立、矛盾）等。在预训练时代，句子匹配任务的标准做法就是将两个句子用 [SEP] 连接起来，然后当成单文本分类任务来做。

<img src="https://img2023.cnblogs.com/blog/1641623/202402/1641623-20240217104746710-874103643.png" alt="fcf444cd-acfa-47cf-8f2b-28c224613f5c" style="zoom: 50%;" />

## 翻译任务

**翻译任务（Translation）**：典型的序列到序列 (sequence-to-sequence, Seq2Seq) 任务，即对于每一个输入序列都会输出一个对应的序列。

## 文本摘要任务

**文本摘要（Text Summarization）**：同样是一个 Seq2Seq 任务，旨在尽可能保留文本语义的情况下将长文本压缩为短文本。文本摘要可以看作是将长文本“翻译”为捕获关键信息的短文本，因此大部分文本摘要模型同样采用 Encoder-Decoder 框架。

## 抽取式问答

**自动问答 (Question Answering, QA)** ：是经典的 NLP 任务，需要模型基于给定的上下文回答问题。根据回答方式的不同可以分为：

1. **抽取式（extractive）问答**：从上下文中截取片段作为回答，类似于序列标注任务；
2. **生成式（generative）问答**：生成一个文本片段作为回答，类似于翻译和摘要任务

抽取式问答模型通常采用纯 Encoder 框架，例如 BERT，它更适用于处理事实性问题，即问题的答案通常就包含在上下文；而生成式问答模型通常采用 Encoder-Decoder 框架（例如 T5、BART），它更适用于处理开放式问题，这些问题的答案通常需要结合上下文语义再进行抽象表达。

<img src="https://img2023.cnblogs.com/blog/1641623/202402/1641623-20240217110155194-58968817.png" alt="fcf444cd-acfa-47cf-8f2b-28c224613f5c" style="zoom: 50%;" />

## 情感分析 Prompting

Prompting 方法的核心思想就是借助模板将问题转换为与预训练任务类似的形式来处理。

例如要判断标题 “American Duo Wins Opening Beach Volleyball Match” 的新闻类别，就可以应用模板 “This is a [MASK] News: x” 将其转换为 “This is a [MASK] News: American Duo Wins Opening Beach Volleyball Match”，然后送入到包含 MLM (Mask Language Modeling) 预训练任务的模型中预测 [MASK] 对应的词，最后将词映射到新闻类别（比如 “Sports” 对应 “体育” 类）。

## 单项选择

**单项选择**：同样算是一种阅读理解任务，同样是一个段落提了多个问题，问题的答案是 4 个所给的候选答案之一，但不一定是段落中的片段。我们将其转换为文本匹配问题，将每个候选答案与段落、问题进行匹配，然后预测时取分数最高的那个。

![img](https://img2023.cnblogs.com/blog/1641623/202402/1641623-20240217104825666-1123251854.png)

这样一来，原来的一个问题就需要拆分为 4 个样本来处理，需要预测 4 次才能做出答案，大大增加了计算量。但让人惊奇的是，这种做法是所有 baseline 中效果最好的，比将所有候选答案拼在一起然后做 4 分类好的多。

## OCR

OCR（Optical Character Recognition）



## 参考资料

1. [Transformers 快速入门](https://transformers.run/nlp/2022-03-24-transformers-note-7.html)
2. [bert4keras 在手，baseline 我有：CLUE 基准代码](https://kexue.fm/archives/8739)