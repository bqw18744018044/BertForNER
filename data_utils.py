# encoding: utf-8
"""
@author: bqw
@time: 2021/7/3 12:19
@file: data_utils.py
@desc: 
"""
from typing import List
from dataclasses import dataclass
from transformers import BertTokenizer


@dataclass
class Example:
    text: List[str]
    label: List[str] = None

    def __post_init__(self):
        if self.label:
            assert len(self.text) == len(self.label)


def read_data(path):
    examples = []
    with open(path, "r", encoding="utf-8") as file:
        text = []
        label = []
        for line in file:
            line = line.strip()
            # 一条文本结束
            if len(line) == 0:
                examples.append(Example(text, label))
                text = []
                label = []
                continue
            text.append(line.split()[0])
            label.append(line.split()[1])
    return examples


def load_tag(path):
    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        tag2id = {tag.strip(): idx for idx, tag in enumerate(lines)}
        id2tag = {idx: tag for tag, idx in tag2id.items()}
    return tag2id, id2tag


tokenizer = BertTokenizer.from_pretrained("E:/pretrained/bert-base-chinese")
tag2id, id2tag = load_tag("./data/tag.txt")


if __name__ == "__main__":
    pass
    """
    examples = read_data("./data/train.txt")
    print(examples[0])
    tag2id, id2tag = load_tag("./data/tag.txt")
    print(tag2id)
    print(id2tag)
    """
