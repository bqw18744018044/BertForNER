# encoding: utf-8
"""
@author: bqw
@time: 2021/7/3 13:21
@file: config.py
@desc: 
"""
from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    use_lstm: bool = field(default=True, metadata={"help": "是否使用LSTM"})
    lstm_hidden_size: int = field(default=500, metadata={"help": "LSTM隐藏层输出的维度"})
    lstm_layers: int = field(default=1, metadata={"help": "堆叠LSTM的层数"})
    lstm_dropout: float = field(default=0.5, metadata={"help": "LSTM的dropout"})
    hidden_dropout: float = field(default=0.5, metadata={"help": "预训练模型输出向量表示的dropout"})
    ner_num_labels: int = field(default=12, metadata={"help": "需要预测的标签数量"})


@dataclass
class OurTrainingArguments:
    checkpoint_dir: str = field(default="./models/checkpoints", metadata={"help": "训练过程中的checkpoints的保存路径"})
    best_dir: str = field(default="./models/best", metadata={"help": "最优模型的保存路径"})
    do_eval: bool = field(default=True, metadata={"help": "是否在训练时进行评估"})
    epoch: int = field(default=5, metadata={"help": "训练的epoch"})
    train_batch_size: int = field(default=8, metadata={"help": "训练时的batch size"})
    eval_batch_size: int = field(default=8, metadata={"help": "评估时的batch size"})


@dataclass
class DataArguments:
    train_file: str = field(default="./data/train.txt", metadata={"help": "训练数据的路径"})
    dev_file: str = field(default="./data/dev.txt", metadata={"help": "测试数据的路径"})
