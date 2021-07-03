# encoding: utf-8
"""
@author: bqw
@time: 2021/7/3 14:00
@file: train.py
@desc: 
"""
from transformers.file_utils import logger, logging
from transformers import TrainingArguments, Trainer
from config import ModelArguments, OurTrainingArguments, DataArguments
from train_utils import NERDataset, collate_fn, ner_metrics
from data_utils import read_data, tokenizer
from model import BertForNER

logger.setLevel(logging.INFO)


def run(model_args: ModelArguments, data_args: DataArguments, args: OurTrainingArguments):
    # 设定训练参数
    training_args = TrainingArguments(output_dir=args.checkpoint_dir,  # 训练中的checkpoint保存的位置
                                      num_train_epochs=args.epoch,
                                      do_eval=args.do_eval,  # 是否进行评估
                                      evaluation_strategy="epoch",  # 每个epoch结束后进行评估
                                      per_device_train_batch_size=args.train_batch_size,
                                      per_device_eval_batch_size=args.eval_batch_size,
                                      load_best_model_at_end=True,  # 训练完成后加载最优模型
                                      metric_for_best_model="f1"  # 评估最优模型的指标，该指标是ner_metrics返回评估指标中的key
                                      )
    # 构建dataset
    train_dataset = NERDataset(read_data(data_args.train_file))
    eval_dataset = NERDataset(read_data(data_args.dev_file))
    # 加载预训练模型
    model = BertForNER.from_pretrained("E:/pretrained/bert-base-chinese", model_args=model_args)
    # 初始化Trainer
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=eval_dataset,
                      tokenizer=tokenizer,
                      data_collator=collate_fn,
                      compute_metrics=ner_metrics)
    # 模型训练
    trainer.train()
    # 训练完成后，加载最优模型并进行评估
    logger.info(trainer.evaluate(eval_dataset))
    # 保存训练好的模型
    trainer.save_model(args.best_dir)


if __name__ == "__main__":
    model_args = ModelArguments(use_lstm=True)
    data_args = DataArguments()
    training_args = OurTrainingArguments(train_batch_size=16, eval_batch_size=32)
    run(model_args, data_args, training_args)