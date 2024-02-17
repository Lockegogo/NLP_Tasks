import numpy as np
from torch.utils.data import DataLoader, Dataset

"""
数据是从 1998 年人民日报语料库中抽取的，原始数据集的格式：字 标签

地 O
点 O
在 O
厦 B-LOC
门 I-LOC
"""

CATEGORIES = ["LOC", "ORG", "PER"]


class PeopleDaily(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = {}
        with open(data_file, "rt", encoding="utf-8") as f:
            for idx, line in enumerate(f.read().split("\n\n")):
                if not line:
                    break
                sentence, labels = "", []
                for i, item in enumerate(line.split("\n")):
                    char, tag = item.split(" ")
                    sentence += char
                    if tag.startswith("B"):
                        # [7, 8, '厦门', 'LOC']
                        labels.append([i, i, char, tag[2:]])  # Remove the B- or I-
                    elif tag.startswith("I"):
                        labels[-1][1] = i
                        labels[-1][2] += char
                # {'sentence': '海钓比赛地点在厦门与金门之间的海域。',
                #  'labels': [[7, 8, '厦门', 'LOC'], [10, 11, '金门', 'LOC']]}
                Data[idx] = {"sentence": sentence, "labels": labels}
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataLoader(args, dataset, tokenizer, batch_size=None, shuffle=False):
    def collote_fn(batch_samples):
        batch_sentence, batch_labels = [], []
        for sample in batch_samples:
            batch_sentence.append(sample["sentence"])
            batch_labels.append(sample["labels"])
        batch_inputs = tokenizer(
            batch_sentence,
            max_length=args.max_seq_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        # 构造 label：[7, 8, '厦门', 'LOC'] -> [0, 0, 0, 0, 0, 0, 0, 1, 2, 0] (如果 B-LOC 对应 1, I-LOC 对应 2)
        batch_label = np.zeros(batch_inputs["input_ids"].shape, dtype=int)
        for s_idx, sentence in enumerate(batch_sentence):
            # 会给句子首尾添加特殊字符([CLS], [SEP] 等)，因此需要将实体标签的位置进行相应的偏移
            encoding = tokenizer(
                sentence, max_length=args.max_seq_length, truncation=True
            )

            # # 将特殊 token 对应的标签设为 -100，这样在计算损失时就会忽略这些 token
            # batch_label[s_idx][0] = -100
            # batch_label[s_idx][len(encoding.tokens())-1:] = -100
            # # 也可以不设为 -100，维持原始的 0 值，然后在计算损失时借助 Attention Mask 来排除填充位置

            for char_start, char_end, _, tag in batch_labels[s_idx]:
                # 使用 char_to_token() 函数将实体标签从原文位置映射到切分后的 token 索引（考虑子词概念）
                token_start = encoding.char_to_token(char_start)
                token_end = encoding.char_to_token(char_end)
                if not token_start or not token_end:
                    continue
                # 使用构建好的映射字典将实体标签转化为实体编号
                batch_label[s_idx][token_start] = args.label2id[f"B-{tag}"]
                batch_label[s_idx][token_start + 1 : token_end + 1] = args.label2id[
                    f"I-{tag}"
                ]
        return {"batch_inputs": batch_inputs, "labels": batch_label}

    return DataLoader(
        dataset,
        batch_size=(batch_size if batch_size else args.batch_size),
        shuffle=shuffle,
        collate_fn=collote_fn,
    )


if __name__ == "__main__":
    # 通过读取文件构造数据集，打印出一个训练样本
    train_data = PeopleDaily("../data/china-people-daily-ner-corpus/example.train")
    valid_data = PeopleDaily("../data/china-people-daily-ner-corpus/example.dev")
    test_data = PeopleDaily("../data/china-people-daily-ner-corpus/example.test")

    print(train_data[0])
    # {'sentence': '海钓比赛地点在厦门与金门之间的海域。', 'labels': [[7, 8, '厦门', 'LOC'], [10, 11, '金门', 'LOC']]}
    # 可以看到我们自定义的数据集成功抽取出了句子中的实体标签，包括实体在原文中的位置以及标签
