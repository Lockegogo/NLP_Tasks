import collections
import json
import logging
import os
import sys

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer, get_scheduler
from torch.optim import AdamW

sys.path.append("../")
from arg import parse_args
from cmrc2018_evaluate import evaluate
from data_loader import CMRC2018, get_dataLoader

from sequence_labeling_extractiveQA.model import BertForExtractiveQA
from tools import seed_everything

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("Model")


def to_device(args, batch_data):
    new_batch_data = {}
    for k, v in batch_data.items():
        if k == "batch_inputs":
            new_batch_data[k] = {k_: v_.to(args.device) for k_, v_ in v.items()}
        else:
            new_batch_data[k] = torch.tensor(v).to(args.device)
    return new_batch_data


def train_loop(args, dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f"loss: {0:>7f}")
    finish_batch_num = epoch * len(dataloader)

    model.train()
    for step, batch_data in enumerate(dataloader, start=1):
        batch_data = to_device(args, batch_data)
        outputs = model(**batch_data)
        loss = outputs[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(
            f"loss: {total_loss/(finish_batch_num + step):>7f}"
        )
        progress_bar.update(1)
    return total_loss


def test_loop(args, dataloader, dataset, model):
    all_example_ids = []
    all_offset_mapping = []
    for batch_data in dataloader:
        all_example_ids += batch_data["example_ids"]
        all_offset_mapping += batch_data["offset_mapping"]
    example_to_features = collections.defaultdict(list)
    for idx, feature_id in enumerate(all_example_ids):
        # idx: chunk 的索引，
        # feature_id: 原始样本的问题 id，如 DEV_0_QUERY_0
        # example_to_features: key 为原始样本的问题 id，value 为该问题对应的所有 chunk 的索引
        # 用于记录每个原始样本被分块成了哪几个新样本，从而合并对应的预测结果(取出最好的预测结果)
        example_to_features[feature_id].append(idx)

    start_logits = []
    end_logits = []
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            batch_data.pop("example_ids")
            batch_data.pop("offset_mapping")
            batch_data = to_device(args, batch_data)
            outputs = model(**batch_data)
            pred_start_logits, pred_end_logit = outputs[1], outputs[2]
            start_logits.append(pred_start_logits.cpu().numpy())
            end_logits.append(pred_end_logit.cpu().numpy())
    # start_logits 包含了多个 batch 的结果，这里使用 np.concatenate 进行拼接，默认在第 0 维
    start_logits = np.concatenate(start_logits)  # (all_chunk_size, seq_len)
    end_logits = np.concatenate(end_logits)  # (all_chunk_size, seq_len)

    theoretical_answers = [
        {"id": dataset[s_idx]["id"], "answers": dataset[s_idx]["answers"]}
        for s_idx in range(len(dataset))
    ]
    predicted_answers = []
    # 对预测结果进行后处理
    for s_idx in tqdm(range(len(dataset))):
        example_id = dataset[s_idx]["id"]
        context = dataset[s_idx]["context"]
        answers = []
        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = all_offset_mapping[feature_index]
            # np.argsort() 返回的是排序后的索引，默认升序，所以最大的概率对应的索引会在最后
            # 最后面的 -1 表示步长，即从后往前取
            start_indexes = np.argsort(start_logit)[-1 : -args.n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -args.n_best - 1 : -1].tolist()
            # 计算 logits 值最高的前 n_best^2 个 token 组合的对
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # 如果 end_index < start_index，说明 end_index 在 start_index 前面，不符合逻辑
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > args.max_answer_length
                    ):
                        continue
                    answers.append(
                        {
                            "start": offsets[start_index][0],
                            "text": context[
                                offsets[start_index][0] : offsets[end_index][1]
                            ],
                            "logit_score": start_logit[start_index]
                            + end_logit[end_index],
                        }
                    )
        # Select the answer with the best score
        if len(answers) > 0:
            # 注意这里不能直接认为最大值=max(start_logit)+max(end_logit)
            # 因为可能会出现不符合逻辑的情况，例如 end_index < start_index
            # 这就是为什么上一步要使用两个 for 循环的原因
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {
                    "id": example_id,
                    "prediction_text": best_answer["text"],
                    "answer_start": best_answer["start"],
                }
            )
        else:
            predicted_answers.append(
                {"id": example_id, "prediction_text": "", "answer_start": 0}
            )
    return evaluate(predicted_answers, theoretical_answers)


def train(args, train_dataset, dev_dataset, model, tokenizer):
    """Train the model"""
    train_dataloader = get_dataLoader(
        args, train_dataset, tokenizer, mode="train", shuffle=True
    )
    dev_dataloader = get_dataLoader(
        args, dev_dataset, tokenizer, mode="valid", shuffle=False
    )
    t_total = len(train_dataloader) * args.num_train_epochs
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        "linear",
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total,
    )
    # Train!
    logger.info("***** Running training *****")
    logger.info(f"Num examples - {len(train_dataset)}")
    logger.info(f"Num Epochs - {args.num_train_epochs}")
    logger.info(f"Total optimization steps - {t_total}")
    with open(os.path.join(args.output_dir, "args.txt"), "wt") as f:
        f.write(str(args))

    total_loss = 0.0
    best_avg_score = 0.0
    for epoch in range(args.num_train_epochs):
        print(f"Epoch {epoch+1}/{args.num_train_epochs}\n" + 30 * "-")
        total_loss = train_loop(
            args, train_dataloader, model, optimizer, lr_scheduler, epoch, total_loss
        )
        metrics = test_loop(args, dev_dataloader, dev_dataset, model)
        F1_score, EM_score, avg_score = metrics["f1"], metrics["em"], metrics["avg"]
        logger.info(
            f"Dev: F1 - {F1_score:0.4f} EM - {EM_score:0.4f} AVG - {avg_score:0.4f}"
        )
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            logger.info(f"saving new weights to {args.output_dir}...\n")
            save_weight = (
                f"epoch_{epoch+1}_dev_f1_{F1_score:0.4f}_em_{EM_score:0.4f}_weights.bin"
            )
            torch.save(model.state_dict(), os.path.join(args.output_dir, save_weight))
    logger.info("Done!")


def test(args, test_dataset, model, tokenizer, save_weights: list):
    test_dataloader = get_dataLoader(
        args, test_dataset, tokenizer, mode="test", shuffle=False
    )
    logger.info("***** Running testing *****")
    for save_weight in save_weights:
        logger.info(f"loading weights from {save_weight}...")
        model.load_state_dict(torch.load(os.path.join(args.output_dir, save_weight)))
        metrics = test_loop(args, test_dataloader, test_dataset, model)
        F1_score, EM_score, avg_score = metrics["f1"], metrics["em"], metrics["avg"]
        logger.info(
            f"Test: F1 - {F1_score:0.4f} EM - {EM_score:0.4f} AVG - {avg_score:0.4f}"
        )


def predict(args, context: str, question: str, model, tokenizer):
    inputs = tokenizer(
        question,
        context,
        max_length=args.max_length,
        truncation="only_second",
        stride=args.stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt",
    )
    chunk_num = inputs["input_ids"].shape[0]
    inputs.pop("overflow_to_sample_mapping")
    offset_mapping = inputs.pop("offset_mapping").numpy().tolist()
    for i in range(chunk_num):
        sequence_ids = inputs.sequence_ids(i)
        offset = offset_mapping[i]
        offset_mapping[i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs = {"batch_inputs": inputs}
    inputs = to_device(args, inputs)
    with torch.no_grad():
        _, pred_start_logits, pred_end_logit = model(**inputs)
        start_logits = pred_start_logits.cpu().numpy()
        end_logits = pred_end_logit.cpu().numpy()
    answers = []
    for feature_index in range(chunk_num):
        start_logit = start_logits[feature_index]
        end_logit = end_logits[feature_index]
        offsets = offset_mapping[feature_index]

        start_indexes = np.argsort(start_logit)[-1 : -args.n_best - 1 : -1].tolist()
        end_indexes = np.argsort(end_logit)[-1 : -args.n_best - 1 : -1].tolist()
        for start_index in start_indexes:
            for end_index in end_indexes:
                if offsets[start_index] is None or offsets[end_index] is None:
                    continue
                if (
                    end_index < start_index
                    or end_index - start_index + 1 > args.max_answer_length
                ):
                    continue
                answers.append(
                    {
                        "start": offsets[start_index][0],
                        "text": context[
                            offsets[start_index][0] : offsets[end_index][1]
                        ],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                )
    if len(answers) > 0:
        best_answer = max(answers, key=lambda x: x["logit_score"])
        return {
            "prediction_text": best_answer["text"],
            "answer_start": best_answer["start"],
        }
    else:
        return {"prediction_text": "", "answer_start": 0}


if __name__ == "__main__":
    args = parse_args()
    if (
        args.do_train
        and os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty."
        )
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.n_gpu = torch.cuda.device_count()
    logger.warning(f"Using {args.device} device, n_gpu: {args.n_gpu}")
    # Set seed
    seed_everything(args.seed)
    args.num_labels = 2
    # Load pretrained model and tokenizer
    logger.info(f"loading pretrained model and tokenizer of {args.model_type} ...")
    config = AutoConfig.from_pretrained(args.model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = BertForExtractiveQA.from_pretrained(
        args.model_checkpoint, config=config, args=args
    ).to(args.device)
    # Training
    if args.do_train:
        train_dataset = CMRC2018(args.train_file)
        dev_dataset = CMRC2018(args.dev_file)
        train(args, train_dataset, dev_dataset, model, tokenizer)
    # Testing
    save_weights = [
        file for file in os.listdir(args.output_dir) if file.endswith(".bin")
    ]
    if args.do_test:
        test_dataset = CMRC2018(args.test_file)
        test(args, test_dataset, model, tokenizer, save_weights)
    # Predicting
    if args.do_predict:
        test_dataset = CMRC2018(args.test_file)
        for save_weight in save_weights:
            logger.info(f"loading weights from {save_weight}...")
            model.load_state_dict(
                torch.load(os.path.join(args.output_dir, save_weight))
            )
            logger.info(f"predicting labels of {save_weight}...")

            results = []
            model.eval()
            for s_idx in tqdm(range(len(test_dataset))):
                sample = test_dataset[s_idx]
                pred_answer = predict(
                    args, sample["context"], sample["question"], model, tokenizer
                )
                results.append(
                    {
                        "id": sample["id"],
                        "title": sample["title"],
                        "context": sample["context"],
                        "question": sample["question"],
                        "answers": sample["answers"],
                        "prediction_text": pred_answer["prediction_text"],
                        "answer_start": pred_answer["answer_start"],
                    }
                )
            with open(
                os.path.join(args.output_dir, save_weight + "_test_data_pred.json"),
                "wt",
                encoding="utf-8",
            ) as f:
                for example_result in results:
                    f.write(json.dumps(example_result, ensure_ascii=False) + "\n")
