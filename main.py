import os
import sys
import argparse
import warnings
from datetime import datetime

import wandb
import torch
import transformers
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Precision
from tqdm import tqdm, trange
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, DeviceStatsMonitor
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme


from data_process import PunDataModule
from model import PunDetModel
import utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_train", type=str, default="True")
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=int, default=[0])
    # parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3, required=False)
    parser.add_argument("--eps", type=float, default=1e-8, required=False, help="AdamW中的eps")
    parser.add_argument("--seed", type=int, default=20220924)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--pretrained_path", type=str, default="/home/chenyang/code/pun/pun_detection/bart-base-chinese")
    parser.add_argument("--train_path", type=str, default="../Data/Dataset/CCL2018_data_3_train.json")
    parser.add_argument("--val_path", type=str, default="../Data/Dataset/CCL2018_data_3_valid.json")
    parser.add_argument("--test_path", type=str, default="../Data/Dataset/CCL2018_data_3_valid.json", required=False)
    parser.add_argument("--checkpoint_path", type=str, default="./saved_models")
    parser.add_argument("--default_root_dir", type=str, default="./")
    parser.add_argument("--label", type=str, default="exp0")
    parser.add_argument('--accumulate_grad_batches', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--warmup_steps', type=int, default=700, help='warm up steps')
    parser.add_argument('--gradient_clip_val', default=None, type=float, required=False)
    parser.add_argument('--log_every_n_steps', default=50, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument("--is_resume", action="store_true", help="是否重新恢复训练")
    parser.add_argument("--resume_checkpoint_path", type=str, default="", required=False, help="训练断点文件的路径")
    parser.add_argument("--num_workers", type=int, default=64)
    parser.add_argument("--deterministic", action="store_false")    # 不输入--deterministic时，默认为True
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--overfit_batches", type=float, default=0.0, help="快速debug")
    parser.add_argument("--precision", type=str, default="16")
    parser.add_argument("--progress_bar_refresh_rate", type=int, default=20)

    args = parser.parse_args()

    return args

def train_model(config, model, datamodule, logger):
    # create your own theme!
    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
        ),
        leave=True,
    )
    checkpoint = ModelCheckpoint(
        dirpath=config.checkpoint_path,
        filename="{epoch:02d}-{val_macro_f1:.4f}",
        save_weights_only=False,
        save_on_train_epoch_end=True,
        monitor="val_macro_f1",
        mode="max",
        save_top_k=3,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    device_stats = DeviceStatsMonitor()
    trainer = pl.Trainer.from_argparse_args(args=config, callbacks=[checkpoint, lr_monitor, device_stats, progress_bar], logger=wandb_logger)
    wandb_logger.watch(model, log="all")
    if config.is_resume:
        trainer.fit(model, datamodule=datamodule, ckpt_path = config.resume_checkpoint_path)
    else:
        trainer.fit(model, datamodule=datamodule)
    print("trainer.checkpoint_callback.best_model_path: ", str(trainer.checkpoint_callback.best_model_path))

    trainer.test(model, datamodule=datamodule, verbose=False)
    

if __name__ == "__main__":
    config = parse_args()

    wandb_logger = WandbLogger(project="Lightning Test", name="test1")
    pl.seed_everything(config.seed)
    torch.set_float32_matmul_precision('high')

    # dt = datetime.now()
    save_path = "/" + config.label
    config.checkpoint_path += save_path

    PunData = PunDataModule(config=config)

    model = PunDetModel(config=config)
    print(config)
    
    train_model(config, model, PunData, wandb_logger)