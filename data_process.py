import os
import json
import sys
import math

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, DataProcessor

# os.environ['TOKENIZERS_PARALLELISM']="false"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))


class PunDataModule(pl.LightningDataModule):

    def __init__(self, config) -> None:
        super().__init__()
        self.train_path = config.train_path
        self.val_path = config.val_path
        self.test_path = config.test_path
        self.tokenizer = BertTokenizer.from_pretrained(config.pretrained_path)
        self.labels = config.num_classes
        self.config = config
        self._load_data()
        

    def read_data(self, filename):
        with open(filename, "r") as f:
            raw_data = json.load(f)
        text, labels = [], []
        label_name = ["homophonic", "homographic", "reverse"]
        for i, name in enumerate(label_name):
            samples = raw_data[name]
            for sample in samples:
                text.append(sample['content'])
                labels.append(i)
        return text, labels
    
    def get_data(self, mode="train"):
        [data, labels] = self.read_data(eval("self.%s_path" % mode))
        data_length = len(data)
        tokenized_data = self.tokenizer(
            data,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        return tokenized_data, labels, data_length

    def _load_data(self):
        print("loading training dataset and validating dataset!")
        self.train_data, self.train_labels, train_data_length = self.get_data(mode="train")
        print("train_length: %d" % train_data_length)
        self.val_data, self.val_labels, val_data_length = self.get_data(mode="val")
        print("valid_length: %d" % val_data_length)
        self.test_data, self.test_labels, test_data_length = self.get_data(mode="test")
        print("test_length: %d" % test_data_length)

    def prepare_data(self):
        ...

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = torch.utils.data.TensorDataset(
                torch.LongTensor(self.train_data["input_ids"]),
                torch.LongTensor(self.train_data["attention_mask"]),
                torch.LongTensor(self.train_labels),
            )
            self.val_dataset = torch.utils.data.TensorDataset(
                torch.LongTensor(self.val_data["input_ids"]),
                torch.LongTensor(self.val_data["attention_mask"]),
                torch.LongTensor(self.val_labels),
            )
        if stage == "test":
            self.test_dataset = torch.utils.data.TensorDataset(
                torch.LongTensor(self.test_data["input_ids"]),
                torch.LongTensor(self.test_data["attention_mask"]),
                torch.LongTensor(self.test_labels),
            )
        self.config.total_steps = math.ceil(self.config.max_epochs / self.config.accumulate_grad_batches * math.ceil(len(self.train_dataset) / self.config.batch_size))
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.config.batch_size, num_workers=self.config.num_workers, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.config.batch_size, num_workers=self.config.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.config.batch_size, num_workers=self.config.num_workers, pin_memory=True)
    
