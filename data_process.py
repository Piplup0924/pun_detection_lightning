import os
import json
import sys
import math
import re

import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, DataProcessor

# os.environ['TOKENIZERS_PARALLELISM']="false"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))


class PunDataModule(pl.LightningDataModule):

    def __init__(self, config) -> None:
        super().__init__()
        self.data_path = config.data_path
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_pretrained_path)
        self.labels = config.num_classes
        self.config = config
        self._load_data()
        
    
    def _get_data(self):
        raw_data = pd.read_csv(self.config.data_path)
        raw_data.dropna(subset=["query", "涨跌语义"], inplace = True)
        text, pun_labels, emotion_labels = raw_data["query"].to_list(), list(map(lambda x: eval(x[0]),raw_data["是否双关"].to_list())), raw_data["涨跌语义"].to_list()
        emotion_dict = {"看好": 0, "看衰": 1, "震荡": 2, "其他": 3}
        emotion_labels = list(map(lambda x: emotion_dict[x], emotion_labels))
        data_length = len(text)
        tokenized_data = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        return tokenized_data, pun_labels, emotion_labels, data_length
    
    def _clean_up(self, x):
        emojipat = re.compile("\[.*?\]")
        apat = re.compile("\{\{\w*_\d*\}\}")
        spanpat = re.compile("<.*?>")
        tagpat = re.compile("#.*?#")
        # x = x.replace("<p>","").replace("</p>", "")
        x = emojipat.sub("", x)
        x = tagpat.sub("", x)
        x = apat.sub("", x)
        x = spanpat.sub("", x)
        return x.strip()

    def _get_pred_data(self):
        raw_data = pd.read_csv(self.config.pred_path)
        raw_data.dropna(subset=["online_content"], inplace=True)
        content = raw_data["online_content"].to_list()
        text = list(filter(bool,list(map(self._clean_up, content))))
        data_length = len(text)
        fake_labels = [0] * data_length
        tokenized_data = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        return tokenized_data, fake_labels, fake_labels, data_length
    
    def _get_pred_data_2(self):
        with open("./results/exp1_results.txt", 'r') as f:
            content = f.readlines()
        text = list(filter(lambda x: len(x) > 2, map(lambda x:x[:-3], content)))
        data_length = len(text)
        fake_labels = [0] * data_length
        tokenized_data = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        return tokenized_data, fake_labels, fake_labels, data_length


    def _load_data(self):
        if self.config.is_train:
            print("loading training dataset and validating dataset!")
            self.data, self.pun_labels, self.emotion_labels, data_length = self._get_data()
            print("data_length: %d" % data_length)
            # self.test_data, self.test_labels, test_data_length = self.get_data(mode="test")
            # print("test_length: %d" % test_data_length)
        elif self.config.is_test:
            print("loading testing dataset!")
            self.data, self.pun_labels, self.emotion_labels, data_length = self._get_data()
            print("data_length: %d" % data_length)
        elif self.config.is_pred:
            print("loading predicting dataset!")
            self.data, self.pun_labels, self.emotion_labels, data_length = self._get_pred_data_2()
            print("data_length: %d" % data_length)

    def prepare_data(self):
        ...

    def setup(self, stage: str):
        if stage == "predict":
            self.pred_dataset = torch.utils.data.TensorDataset(
                torch.LongTensor(self.data["input_ids"]),
                torch.LongTensor(self.data["attention_mask"]),
                torch.LongTensor(self.pun_labels),
                torch.LongTensor(self.emotion_labels)
            )
        elif stage == "test":
            self.dataset = torch.utils.data.TensorDataset(
                torch.LongTensor(self.data["input_ids"]),
                torch.LongTensor(self.data["attention_mask"]),
                torch.LongTensor(self.pun_labels),
                torch.LongTensor(self.emotion_labels)
            )
            self.train_dataset, self.val_dataset = random_split(self.dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(20220924))
            self.test_dataset = self.val_dataset
        else:
            self.dataset = torch.utils.data.TensorDataset(
                torch.LongTensor(self.data["input_ids"]),
                torch.LongTensor(self.data["attention_mask"]),
                torch.LongTensor(self.pun_labels),
                torch.LongTensor(self.emotion_labels)
            )
            self.train_dataset, self.val_dataset = random_split(self.dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(20220924))
            self.test_dataset = self.val_dataset

            self.config.total_steps = math.ceil(self.config.max_epochs / self.config.accumulate_grad_batches * math.ceil(len(self.train_dataset) / self.config.batch_size))
        
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.config.batch_size, num_workers=self.config.num_workers, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.config.batch_size, num_workers=self.config.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.config.batch_size, num_workers=self.config.num_workers, pin_memory=True)
    
    def predict_dataloader(self):
        return DataLoader(self.pred_dataset, batch_size = self.config.pred_batch_size, num_workers=self.config.num_workers, pin_memory=True)
