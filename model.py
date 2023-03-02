import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import transformers
from  torchmetrics import Accuracy, Precision, Recall, F1Score
from transformers import BartForSequenceClassification, BertTokenizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

class PunDetModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.predict_step_preds, self.predict_step_texts = [], []
        self.tokenizer = BertTokenizer.from_pretrained(config.pretrained_path)
        self.model = BartForSequenceClassification.from_pretrained(config.pretrained_path, num_labels = config.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=3)
        self.val_precision = Precision(task="multiclass", average="macro", num_classes=3)
        self.val_recall = Recall(task="multiclass", average="macro", num_classes=3)
        self.val_macro_f1 = F1Score(task="multiclass", average="macro", num_classes=3)
        self.test_acc = Accuracy(task="multiclass", num_classes=3)
        self.test_precision = Precision(task="multiclass", average="macro", num_classes=3)
        self.test_recall = Recall(task="multiclass", average="macro", num_classes=3)
        self.test_macro_f1 = F1Score(task="multiclass", average="macro", num_classes=3)

    def forward(self, input_ids, labels, attention_mask = None):
        """
        Params input_ids: [batch_size, seq_len]
        Params attention_mask: [batch_size, seq_len]

        Return outputs: [batch_size, seq_len, num_labels]
        """
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)

        return outputs

    def training_step(self, batch, batch_idx):
        x, a, y = batch
        outputs = self(input_ids = x, attention_mask = a, labels = y)
        logits = outputs.logits
        loss = outputs.loss
        self.training_step_outputs.append(logits)
        self.log("train_loss", loss, on_step=True, logger=True)
        return loss
    
    def on_train_epoch_end(self):
        # all_preds = torch.stack(self.training_step_outputs)
        ...
        self.training_step_outputs.clear()
    
    def validation_step(self, batch, batch_idx):
        x, a, y = batch
        outputs = self(input_ids = x, attention_mask = a, labels = y)
        logits = outputs.logits     # [batch_size, num_cls]
        loss = outputs.loss
        self.validation_step_outputs.append(logits)
        pred = torch.argmax(F.softmax(logits, dim = 1), dim = 1)
        self.val_acc(pred, y)
        self.val_precision(pred, y)
        self.val_recall(pred, y)
        self.val_macro_f1(pred, y)
        metrics = {"val_loss": loss, "val_acc": self.val_acc, "val_precision": self.val_precision, "val_recall": self.val_recall}
        self.log_dict(metrics, on_epoch=True, logger=True)
        self.log("val_macro_f1", self.val_macro_f1, on_step=False, on_epoch=True, prog_bar=True)
        return logits

    def on_validation_epoch_end(self):
        # all_preds = torch.stack(self.validation_step_outputs)
        self.validation_step_outputs.clear()
    
    # def test_step(self, batch, batch_idx):
        # x, a, y = batch
        # outputs = self(input_ids = x, attention_mask = a, labels = y)
        # logits = outputs.logits
        # pred = torch.argmax(F.softmax(logits, dim = 1), dim = 1)
        # self.test_acc(pred, y)
        # self.test_precision(pred, y)
        # self.test_recall(pred, y)
        # self.test_macro_f1(pred, y)
        # metrics = {"test_acc": self.test_acc, "test_precision": self.test_precision, "test_recall": self.test_recall, "test_macro_f1": self.test_macro_f1}
        # self.log_dict(metrics, on_epoch=True, logger=True)
        # return logits
    
    def predict_step(self, batch, batch_idx, dataloader_idx = 0):
        x, a, y = batch     # [batch_size, max_seq_len], [batch_size, max_seq_len], [batch_size, num_cls]
        outputs = self(input_ids = x, attention_mask = a, labels = y)
        logits = outputs.logits
        text = self.tokenizer.batch_decode(x, skip_special_tokens=True)
        pred = torch.argmax(F.softmax(logits, dim = 1), dim = 1)
        self.predict_step_preds.append(pred.tolist())
        self.predict_step_texts.append(text)
        return pred
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, eps=self.config.eps)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": transformers.get_linear_schedule_with_warmup(
            #             optimizer=optimizer, 
            #             num_training_steps = self.config.total_steps
            #         ),
            #     "monitor": "metric_to_track",
            #     "interval": "step"
            # }
        }

if __name__ == "__main__":
    ...