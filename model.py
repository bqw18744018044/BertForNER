# encoding: utf-8
"""
@author: bqw
@time: 2021/7/3 13:16
@file: model.py
@desc: 
"""
import torch
import torch.nn as nn

from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import TokenClassifierOutput


class BertForNER(BertPreTrainedModel):
    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        if "model_args" in model_kargs:
            model_args = model_kargs["model_args"]
            # 必须将额外的参数更新至self.config中，不然再使用from_pretrianed方法加载训练好的模型时会出错
            self.config.__dict__.update(model_args.__dict__)
        self.num_labels = self.config.ner_num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(self.config.hidden_dropout)
        self.lstm = nn.LSTM(self.config.hidden_size,
                            self.config.lstm_hidden_size,
                            num_layers=self.config.lstm_layers,
                            dropout=self.config.lstm_dropout,
                            bidirectional=True,
                            batch_first=True)
        if self.config.use_lstm:
            self.classifier = nn.Linear(self.config.lstm_hidden_size * 2, self.num_labels)
        else:
            self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            pos=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = self.dropout(outputs[0])
        if self.config.use_lstm:
            sequence_output, _ = self.lstm(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )