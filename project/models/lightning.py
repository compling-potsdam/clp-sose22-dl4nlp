from datetime import datetime
from typing import Any
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from argparse import ArgumentParser

import torchmetrics as tm
from torchvision import datasets
from torchvision.transforms import ToTensor


class LitNeuralNetwork(pl.LightningModule):

    # noinspection PyUnusedLocal
    def __init__(self, data_dir, batch_size, hidden_size, num_classes, **kwargs):
        super(LitNeuralNetwork, self).__init__()
        self.data_dir = data_dir
        self.save_hyperparameters()

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hparams.hidden_size, self.hparams.num_classes)
        )

        self.accuracy = tm.Accuracy()
        self.train_data, self.val_data = None, None

    def prepare_data(self):
        datasets.FashionMNIST(root=self.data_dir, train=True, download=True)
        datasets.FashionMNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_data = datasets.FashionMNIST(root=self.data_dir, train=True, transform=ToTensor())
            self.val_data = datasets.FashionMNIST(root=self.data_dir, train=False, transform=ToTensor())

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-6)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
        return [optimizer], [scheduler]

    def forward(self, x) -> Any:
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        predictions = torch.argmax(logits, dim=1)  # B x V x L -> B x L
        self.accuracy.update(predictions, y)
        return loss

    def validation_epoch_end(self, step_outputs) -> None:
        # step_or_epoch = self.global_step
        step_or_epoch = self.current_epoch + 1
        self.log_dict({"step": step_or_epoch, "val/acc": self.accuracy.compute()})

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.hparams.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.hparams.batch_size, shuffle=False, num_workers=0)

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass

    def configure_callbacks(self):
        monitor_metric = "val/acc"  # also adjust checkpoint filename!
        return [
            pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
            pl.callbacks.EarlyStopping(monitor=monitor_metric, mode="max", patience=20, verbose=True),
            pl.callbacks.ModelCheckpoint(filename="model-epoch={epoch:02d}-val-acc={val/acc:.2f}",
                                         monitor=monitor_metric, mode="max", save_top_k=3, verbose=True,
                                         auto_insert_metric_name=False)
        ]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitNeuralNetwork")
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--hidden_size", type=int, default=512)
        parser.add_argument("--num_classes", type=int, default=10)
        return parent_parser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpts_root", default="/Users/philippsadler/Opts/Git/clp-sose22-dl4nlp/logs")
    parser.add_argument("--data_dir", type=str, default="/Users/philippsadler/Opts/Git/clp-sose22-dl4nlp/data")
    parser = LitNeuralNetwork.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    # tensorboard filter: (metrics)|(loss)|(grad_2.0_norm_total_epoch)
    # tensorboard filter: train|val
    project_name = "fashion-mnist"
    print("Checkpoint directory:", args.ckpts_root)
    tb_logger = TensorBoardLogger(
        save_dir=args.ckpts_root,
        name=datetime.now().strftime(f"{project_name}/M%mD%d"),  # this is used for ModelCheckpoint path !
        version=datetime.now().strftime("%H%M%S"),  # this is used for ModelCheckpoint path !
    )
    dict_args = vars(args)
    model = LitNeuralNetwork(**dict_args)
    trainer = pl.Trainer.from_argparse_args(args, gpus=None,
                                            max_epochs=3,
                                            # max_steps=10,
                                            log_every_n_steps=1,  # step = batch
                                            logger=tb_logger,
                                            val_check_interval=.2,  # perform validation more often
                                            weights_save_path=args.ckpts_root,
                                            gradient_clip_val=10
                                            )
    trainer.fit(model)
