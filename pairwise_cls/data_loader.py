import json

from torch.utils.data import DataLoader, Dataset


class AFQMC(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = {}
        with open(data_file, "rt", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


import json

# 如果数据集非常巨大，难以一次性加载到内存中，我们也可以继承 IterableDataset 类构建迭代型数据集
from torch.utils.data import IterableDataset


class IterableAFQMC(IterableDataset):
    def __init__(self, data_file):
        self.data_file = data_file

    def __iter__(self):
        with open(self.data_file, "rt") as f:
            for line in f:
                sample = json.loads(line.strip())
                yield sample


def get_dataLoader(args, dataset, tokenizer, batch_size=None, shuffle=False):
    # 手工编写 Dataloader 的批处理函数
    def collote_fn(batch_samples):
        batch_sentence_1, batch_sentence_2, batch_label = [], [], []
        for sample in batch_samples:
            batch_sentence_1.append(sample["sentence1"])
            batch_sentence_2.append(sample["sentence2"])
            batch_label.append(int(sample["label"]))
        batch_inputs = tokenizer(
            batch_sentence_1,
            batch_sentence_2,
            max_length=args.max_seq_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return {"batch_inputs": batch_inputs, "labels": batch_label}

    return DataLoader(
        dataset,
        batch_size=(batch_size if batch_size else args.batch_size),
        shuffle=shuffle,
        collate_fn=collote_fn,
    )
