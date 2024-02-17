import json

import torch
from torch.utils.data import DataLoader, Dataset

"""
数据是从 translation2019zh 语料库中抽取的，原始数据集的格式：
{"english": "In Italy, there is no real public pressure for a new, fairer tax system.",
 "chinese": "在意大利，公众不会真的向政府施压，要求实行新的、更公平的税收制度。"}

我们从五百多万条样本中抽取训练集中的前 22 万条数据，并从中划分出 2 万条作为验证集，原本的验证集作为测试集
"""

MAX_DATASET_SIZE = 220000
TRAIN_SET_SIZE = 200000
VALID_SET_SIZE = 20000


class TRANS(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        # 将数据组织成带序号的字典形式
        Data = {}
        with open(data_file, "rt", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= MAX_DATASET_SIZE:
                    break
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataLoader(args, dataset, model, tokenizer, batch_size=None, shuffle=False):
    # 对于每个 batch，调用 collote_cn 将其处理成模型可以读入的格式
    def collote_fn(batch_samples):
        batch_inputs, batch_targets = [], []
        for sample in batch_samples:
            batch_inputs.append(sample["chinese"])
            batch_targets.append(sample["english"])
        # 返回的 batch_data 是字典，包含的 key 为：input_ids，attention_mask，token_type_ids
        batch_data = tokenizer(
            batch_inputs,
            padding=True,
            max_length=args.max_input_length,
            truncation=True,
            return_tensors="pt",
        )
        # 上下文管理器
        with tokenizer.as_target_tokenizer():
            # 对 label 来说，显然不需要 attention_mask，token_type_ids
            labels = tokenizer(
                batch_targets,
                padding=True,
                max_length=args.max_target_length,
                truncation=True,
                return_tensors="pt",
            )["input_ids"]
            # 移位操作：对 seq2seq 任务来说，我们还需要标签序列的移位作为 Decoder 的输入，即在标签序列的开头添加一个特殊的“序列起始符”
            batch_data[
                "decoder_input_ids"
            ] = model.prepare_decoder_input_ids_from_labels(labels)
            # Marian 模型会在分词结果的结尾加上特殊 token "</s>"：['它们', '看起来', '真的很', '可爱', '。', '</s>']
            # 因此这里通过 tokenizer.eos_token_id 定位其在 token ID 序列中的索引
            # 然后将其后的 pad 字符设置为 -100
            # torch.where() 返回两个张量，表示满足条件的元素的索引，第一个张量是行索引，第二个张量是列索引。这里取出列索引
            end_token_index = torch.where(labels == tokenizer.eos_token_id)[1]
            for idx, end_idx in enumerate(end_token_index):
                labels[idx][end_idx + 1 :] = -100
            batch_data["labels"] = labels

        # 模型 AutoModelForSeq2Seq2SeqLM 可接受的参数为：一个包含 input_ids，attention_mask，decoder_input_ids，labels 的字典
        # 所以将 batch_data 构造成这样的字典
        return batch_data

    return DataLoader(
        dataset,
        batch_size=(batch_size if batch_size else args.batch_size),
        shuffle=shuffle,
        collate_fn=collote_fn,
    )
