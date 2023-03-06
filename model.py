import os
import sys
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import transformers
from  torchmetrics import Accuracy, Precision, Recall, F1Score
from transformers import BartForSequenceClassification, BertTokenizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from utils import create_dir

class PunDetModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.epoch = 0
        self.config = config
        self.training_step_outputs = []
        self.validation_step_preds, self.validation_step_labels = [], []
        self.test_step_preds, self.test_step_labels = [], []
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

        Return:
        outputs: [batch_size, seq_len, num_labels]
        """
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)

        return outputs
    
    def _count_chunks(self, pred, y):
        """
        Params pred: a list of predictions
        Params y: a list of true labels

        Return:
        correct_counts：a dict, number of correctly identified samples per type 每种类型预测正确的sample数量
        true_counts: a dict, number of true samples per type   每种类型标签为正的sample数量
        pred_counts: a dict, number of identified samples per type  每种类型预测的数量
        """
        correct_counts, true_counts, pred_counts = defaultdict(int), defaultdict(int), defaultdict(int)

        for true_tag, pred_tag in zip(y, pred):
            if true_tag == pred_tag:
                correct_counts[true_tag] += 1
            true_counts[true_tag] += 1
            pred_counts[pred_tag] += 1
        
        return correct_counts, true_counts, pred_counts

    def on_train_epoch_start(self) -> None:
        self.epoch += 1

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
    
    def _calc_metrics(self, tp, p, t, percent=True):
        """
        Params tp: 所有预测正确的正样本
        Params p: 所有预测为正的样本
        Params t: 所有正样本

        compute overall precision, recall and F1
        if percent is True, return 100 * original decimal value
        """
        precision = tp / p if p else 0
        recall = tp / t if t else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        if percent:
            return 100 * precision, 100 * recall, 100 * f1
        else:
            return precision, recall, f1

    def _get_result(self, correct_counts, true_counts, pred_counts):
        pun_types = defaultdict(int, {0: "homophonic", 1: "homographic", 2: "others"})

        for t in pun_types.keys():
            precision, recall, f1 = self._calc_metrics(correct_counts[t], pred_counts[t], true_counts[t])
            # with open(os.path.join(self.config.checkpoint_path, self.config.label, "val_log.txt"), "w") as fout:
            #     fout.write("%17s: Precision: %6.2f%%; Recall: %6.2f%%; F1: %6.2f%%  (%d & %d) = %d\n" % (pun_types[t], precision, recall, f1, pred_counts[t], true_counts[t], correct_counts[t]))
            print("%17s: " % pun_types[t], end = '')
            print("Precision: %6.2f%%; Recall: %6.2f%%; F1: %6.2f%%" % (precision, recall, f1), end="")
            print("  (%d & %d) = %d" % (pred_counts[t], true_counts[t], correct_counts[t]))


    def validation_step(self, batch, batch_idx):
        x, a, y = batch
        outputs = self(input_ids = x, attention_mask = a, labels = y)
        logits = outputs.logits     # [batch_size, num_cls]
        loss = outputs.loss
        self.validation_step_outputs.append(logits)
        pred = torch.argmax(F.softmax(logits, dim = 1), dim = 1)
        self.validation_step_preds.extend(pred.cpu().numpy())
        self.validation_step_labels.extend(y.cpu().numpy())
        self.val_acc(pred, y)
        self.val_precision(pred, y)
        self.val_recall(pred, y)
        self.val_macro_f1(pred, y)
        metrics = {"val_loss": loss, "val_acc": self.val_acc, "val_precision": self.val_precision, "val_recall": self.val_recall}
        self.log_dict(metrics, on_epoch=True, logger=True)
        self.log("val_macro_f1", self.val_macro_f1, on_step=False, on_epoch=True, prog_bar=True)
        return logits

    def on_validation_epoch_end(self) -> None:
        # all_preds = torch.stack(self.validation_step_outputs)
        correct_counts, true_counts, pred_counts = self._count_chunks(self.validation_step_preds, self.validation_step_labels)
        self._get_result(correct_counts, true_counts, pred_counts)
        self.validation_step_preds.clear()
        self.validation_step_labels.clear()

    
    def test_step(self, batch, batch_idx):
        x, a, y = batch
        outputs = self(input_ids = x, attention_mask = a, labels = y)
        logits = outputs.logits
        pred = torch.argmax(F.softmax(logits, dim = 1), dim = 1)
        self.test_step_preds.extend(pred.cpu().numpy())
        self.test_step_labels.extend(y.cpu().numpy())
    
    def on_test_epoch_end(self) -> None:
        correct_counts, true_counts, pred_counts = self._count_chunks(self.test_step_preds, self.test_step_labels)
        self._get_result(correct_counts, true_counts, pred_counts)
        self.test_step_preds.clear()
        self.test_step_labels.clear()
        

    def predict_step(self, batch, batch_idx, dataloader_idx = 0):
        x, a, y = batch     # [batch_size, max_seq_len], [batch_size, max_seq_len], [batch_size,]
        outputs = self(input_ids = x, attention_mask = a, labels = y)
        logits = outputs.logits
        text = self.tokenizer.batch_decode(x, skip_special_tokens=True)
        pred = torch.argmax(F.softmax(logits, dim = 1), dim = 1)
        self.predict_step_preds.extend(pred.tolist())
        self.predict_step_texts.extend(text)
        return pred
    
    def on_predict_epoch_end(self, results) -> None:
        assert len(self.predict_step_preds) == len(self.predict_step_texts)
        create_dir("./results")
        with open(os.path.join("./results/", self.config.label+"_results.txt"), "w") as f:
            for i, text in enumerate(self.predict_step_texts):
                f.write("".join(text.split(" ")) + "\t" + str(self.predict_step_preds[i]) + "\n")
        print("The num of predictions is " + str(len(self.predict_step_preds)))

    
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