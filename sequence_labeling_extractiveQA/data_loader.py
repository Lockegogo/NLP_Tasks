import json

import torch
from torch.utils.data import DataLoader, Dataset

"""
数据是哈工大讯飞联合实验室构建的中文阅读理解语料库 CMRC 2018 ，原始数据集的格式：

{
 "context": "《战国无双3》（）是由光荣和ω-force开发的战国无双系列的正统第三续作。本作以三大故事为主轴，分别是以武田信玄等人为主的《关东三国志》，织田信长等人为主的《战国三杰》，石田三成等人为主的《关原的年轻武者》，丰富游戏内的剧情。此部份专门介绍角色，欲知武器情报、奥义字或擅长攻击类型等，请至战国无双系列1.由于乡里大辅先生因故去世，不得不寻找其他声优接手。从猛将传 and Z开始。2.战国无双 编年史的原创男女主角亦有专属声优。此模式是任天堂游戏谜之村雨城改编的新增模式。...",
 "qas": [{
     "question": "《战国无双3》是由哪两个公司合作开发的？",
     "id": "DEV_0_QUERY_0",
     "answers": [{
         "text": "光荣和ω-force",
         "answer_start": 11
     }, {...}]
 }, {...}]

注意一个问题可能对应着多个参考答案，在训练时我们任意选择其中一个作为标签
在验证/测试时，我们则将预测答案和所有参考答案都送入打分函数来评估模型的性能
"""


class CMRC2018(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = {}
        with open(data_file, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            idx = 0
            for article in json_data["data"]:
                title = article["title"]
                context = article["paragraphs"][0]["context"]
                for question in article["paragraphs"][0]["qas"]:
                    q_id = question["id"]
                    ques = question["question"]
                    text = [ans["text"] for ans in question["answers"]]
                    answer_start = [ans["answer_start"] for ans in question["answers"]]
                    Data[idx] = {
                        "id": q_id,
                        "title": title,
                        "context": context,
                        "question": ques,
                        "answers": {"text": text, "answer_start": answer_start},
                    }
                    idx += 1
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataLoader(
    args, dataset, tokenizer, mode="train", batch_size=None, shuffle=False
):
    assert mode in ["train", "valid", "test"]

    def train_collote_fn(batch_samples):
        batch_question, batch_context, batch_answers = [], [], []
        for sample in batch_samples:
            batch_question.append(sample["question"])  # batch_question = ["问题1", "问题2"]
            batch_context.append(sample["context"])  # batch_context = ["原文1", "原文2"]
            batch_answers.append(sample["answers"])
            # batch_answers = [{"text": ["答案1"], "answer_start": [0]}, {"text": ["答案2"], "answer_start": [0]}]

        batch_inputs = tokenizer(
            batch_question,  # [CLS] question [SEP] context [SEP]
            batch_context,
            max_length=args.max_length,
            truncation="only_second",  # 仅截断第二个句子（原文）
            stride=args.stride,  # 步长，用于控制分块时的重叠
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )  # 分块操作，返回的是一个字典，包含的 key 为：input_ids，attention_mask，token_type_ids, offset_mapping, overflow_to_sample_mapping

        offset_mapping = batch_inputs.pop(
            "offset_mapping"
        )  # (all_chunk_size, seq_len, 2)，注意这里不是 batch_size，而是所有分块后的总数
        sample_mapping = batch_inputs.pop(
            "overflow_to_sample_mapping"
        )  # 一维列表，长度为 all_chunk_size，代表每个 chunk 对应的原始样本的索引

        start_positions = []
        end_positions = []
        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_mapping[i]  # 第 i 个 chunk 对应的原始样本的索引
            answer = batch_answers[sample_idx]  # 第 i 个 chunk 对应的原始样本的答案
            start_char = answer["answer_start"][0]  # 答案在原文中的起始位置
            end_char = answer["answer_start"][0] + len(answer["text"][0])  # 答案在原文中的结束位置
            sequence_ids = batch_inputs.sequence_ids(i)  # 第 i 个 chunk 的 sequence_ids
            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            # 结束上述循环时，sequence_ids[idx] == 1，此时 idx 对应的是 context(文本内容) 的第一个 token
            # [CLS] question [SEP] context [SEP]
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1
            # If the answer is not fully inside the context, label is (0, 0)
            if (
                # 如果答案的起始位置小于 context 的起始位置，或者答案的结束位置大于 context 的结束位置，那么答案就不在 context 中
                offset[context_start][0] > start_char
                or offset[context_end][1] < end_char
            ):
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                # 将 idx 移动到 start_char 所在的位置
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                # 将 idx 移动到 end_char 所在的位置
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)
        return {
            "batch_inputs": batch_inputs,  # 字典，key 对应的 value 的 shape 为 (all_chunk_size, max_seq_len)
            "start_positions": start_positions,  # 一维列表，长度为 all_chunk_size，代表每个 chunk 对应的答案在原文中的起始位置
            "end_positions": end_positions,  # 一维列表，长度为 all_chunk_size，代表每个 chunk 对应的答案在原文中的结束位置
        }

    def test_collote_fn(batch_samples):
        batch_id, batch_question, batch_context = [], [], []
        for sample in batch_samples:
            batch_id.append(sample["id"])
            batch_question.append(sample["question"])
            batch_context.append(sample["context"])
        batch_inputs = tokenizer(
            batch_question,
            batch_context,
            max_length=args.max_length,
            truncation="only_second",
            stride=args.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )
        offset_mapping = batch_inputs.pop("offset_mapping").numpy().tolist()
        sample_mapping = batch_inputs.pop("overflow_to_sample_mapping")
        example_ids = []
        for i in range(len(batch_inputs["input_ids"])):
            sample_idx = sample_mapping[i]
            example_ids.append(batch_id[sample_idx])

            sequence_ids = batch_inputs.sequence_ids(i)
            offset = offset_mapping[i]
            offset_mapping[i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]
        return {
            "batch_inputs": batch_inputs,  # 字典，key 对应的 value 的 shape 为 (all_chunk_size, max_seq_len)
            "offset_mapping": offset_mapping,  # (all_chunk_size, seq_len, 2)
            "example_ids": example_ids,  # 一维列表，长度为 all_chunk_size，代表每个 chunk 对应的原始样本的问题 id，如 DEV_0_QUERY_0
        }

    if mode == "train":
        collote_fn = train_collote_fn
    else:
        collote_fn = test_collote_fn

    return DataLoader(
        dataset,
        batch_size=(batch_size if batch_size else args.batch_size),
        shuffle=shuffle,
        collate_fn=collote_fn,
    )
