# encoding: utf-8
"""
@author: bqw
@time: 2021/7/3 12:47
@file: train_utils.py
@desc: 
"""
import torch
import numpy as np

from torch import Tensor
from model import BertForNER
from typing import List, Dict
from data_utils import Example
from config import ModelArguments
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from data_utils import tokenizer, tag2id, read_data
from transformers.trainer_utils import EvalPrediction
from sklearn.metrics import f1_score, precision_score, recall_score


class NERDataset(Dataset):
    def __init__(self, examples: List[Example], max_length=128):
        self.max_length = 512 if max_length > 512 else max_length
        self.texts = [torch.LongTensor(tokenizer.encode(example.text[: self.max_length - 2])) for example in examples]
        self.labels = []
        for example in examples:
            label = example.label
            label = [tag2id["<start>"]] + [tag2id[l] for l in label][: self.max_length - 2] + [tag2id["<eos>"]]
            self.labels.append(torch.LongTensor(label))
        assert len(self.texts) == len(self.labels)
        for text, label in zip(self.texts, self.labels):
            assert len(text) == len(label)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        return {
            "input_ids": self.texts[item],
            "labels": self.labels[item]
        }


def collate_fn(features) -> Dict[str, Tensor]:
    batch_input_ids = [feature["input_ids"] for feature in features]
    batch_labels = [feature["labels"] for feature in features]
    batch_attentiton_mask = [torch.ones_like(feature["input_ids"]) for feature in features]
    # padding
    batch_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    batch_labels = pad_sequence(batch_labels, batch_first=True, padding_value=tag2id["<pad>"])
    batch_attentiton_mask = pad_sequence(batch_attentiton_mask, batch_first=True, padding_value=0)
    assert batch_input_ids.shape == batch_labels.shape
    return {"input_ids": batch_input_ids, "labels": batch_labels, "attention_mask": batch_attentiton_mask}


def ner_metrics(eval_output: EvalPrediction) -> Dict[str, float]:
    """
    该函数是回调函数，Trainer会在进行评估时调用该函数
    """
    preds = eval_output.predictions
    preds = np.argmax(preds, axis=-1).flatten()
    labels = eval_output.label_ids.flatten()
    # labels为0表示为<pad>，因此计算时需要去掉该部分
    mask = labels != 0
    preds = preds[mask]
    labels = labels[mask]
    metrics = dict()
    metrics["f1"] = f1_score(labels, preds, average="macro")
    metrics["precision"] = precision_score(labels, preds, average="macro")
    metrics["recall"] = recall_score(labels, preds, average="macro")
    return metrics


if __name__ == "__main__":
    examples = read_data("./data/train.txt")
    dataset = NERDataset(examples)
    print(dataset[0])
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(dataloader))
    print(batch)
    model_args = ModelArguments(use_lstm=False)
    model = BertForNER.from_pretrained("E:/pretrained/bert-base-chinese", model_args=model_args)
    output = model(**batch)
    print(output)

