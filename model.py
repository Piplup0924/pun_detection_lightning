import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import transformers
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
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def on_train_epoch_end(self):
        # all_preds = torch.stack(self.training_step_outputs)
        self.print("")
        self.training_step_outputs.clear()
    
    def validation_step(self, batch, batch_idx):
        x, a, y = batch
        outputs = self(input_ids = x, attention_mask = a, labels = y)
        logits = outputs.logits     # [batch_size, num_cls]
        loss = outputs.loss
        self.validation_step_outputs.append(logits)
        pred = torch.argmax(F.softmax(logits, dim = 1), dim = 1)
        accuracy, precision, recall, macro_f1 = self._shared_log_step(y, pred)
        metrics = {"val_loss": loss, "val_acc": accuracy, "val_precision": precision, "val_recall": recall, "val_macro_f1": macro_f1}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return logits
    
    def _shared_log_step(self, y, pred):
        y_true = y.cpu().numpy()
        y_pred = pred.cpu().numpy()
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        return accuracy, precision, recall, macro_f1

    def on_validation_epoch_end(self):
        # all_preds = torch.stack(self.validation_step_outputs)
        self.print('')
        self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        x, a, y = batch
        outputs = self(input_ids = x, attention_mask = a, labels = y)
        logits = outputs.logits
        self.test_step_outputs.append(logits)
        pred = torch.argmax(F.softmax(logits, dim = 1), dim = 1)
        accuracy, precision, recall, macro_f1 = self._shared_log_step(y, pred)
        metrics = {"test_acc": accuracy, "test_precision": precision, "test_recall": recall, "test_macro_f1": macro_f1}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return logits
    
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
            "lr_scheduler": {
                "scheduler": transformers.get_linear_schedule_with_warmup(
                        optimizer=optimizer, 
                        num_warmup_steps=self.config.warmup_steps, 
                        num_training_steps = self.config.total_steps
                    ),
                "monitor": "metric_to_track",
                "interval": "step"
            }
        }

if __name__ == "__main__":
    ...