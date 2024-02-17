import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

"""
数据是从 LCSTS 数据集中抽取的，原始数据集的格式：title!=!content

媒体融合关键是以人为本!=!受众在哪里，媒体就应该在哪里，媒体的体制、内容、技术就应该向哪里转变。媒体融合关键是以人为本，即满足大众的信息需求，为受众提供更优质的服务。这就要求媒体在融合发展的过程中，既注重技术创新，又注重用户体验。
"""

MAX_DATASET_SIZE = 200000


class LCSTS(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = {}
        with open(data_file, "rt", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= MAX_DATASET_SIZE:
                    break
                items = line.strip().split("!=!")
                assert len(items) == 2
                Data[idx] = {"title": items[0], "content": items[1]}
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataLoader(args, dataset, model, tokenizer, batch_size=None, shuffle=False):
    def collote_fn(batch_samples):
        batch_inputs, batch_targets = [], []
        for sample in batch_samples:
            batch_inputs.append(sample["content"])
            batch_targets.append(sample["title"])
        batch_data = tokenizer(
            batch_inputs,
            padding=True,
            max_length=args.max_input_length,
            truncation=True,
            return_tensors="pt",
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch_targets,
                padding=True,
                max_length=args.max_target_length,
                truncation=True,
                return_tensors="pt",
            )["input_ids"]
            batch_data[
                "decoder_input_ids"
            ] = model.prepare_decoder_input_ids_from_labels(labels)
            end_token_index = torch.where(labels == tokenizer.eos_token_id)[1]
            for idx, end_idx in enumerate(end_token_index):
                labels[idx][end_idx + 1 :] = -100
            batch_data["labels"] = labels

        return batch_data

    return DataLoader(
        dataset,
        batch_size=(batch_size if batch_size else args.batch_size),
        shuffle=shuffle,
        collate_fn=collote_fn,
    )


if __name__ == "__main__":
    # 通过读取文件构造数据集，打印出一个训练样本
    train_data = LCSTS("../data/lcsts/train.txt")
    valid_data = LCSTS("../data/lcsts/dev.txt")
    test_data = LCSTS("../data/lcsts/test.txt")

    print(train_data[0])
    # {'title': '修改后的立法法全文公布', 'content': '新华社受权于18日全文播发修改后的《中华人民共和国立法法》，修改后的立法法分为“总则”“法律”“行政法规”“地方性法规、自治条例和单行条例、规章”“适用与备案审查”“附则”等6章，共计105条。'}