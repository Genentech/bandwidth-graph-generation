import abc
import math

import torch
from torch import optim
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from pytorch_lightning import LightningModule


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def configure_optimizer(parameters, lr: float, wd: float, iterations: int):
    opt = optim.AdamW(parameters, lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iterations)
    return {
        "optimizer": opt,
        "lr_scheduler": scheduler,
    }


class SinusoidalPositionEmbeddings(nn.Module):
    """Used for time embeddings of diffusion model"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor):
        """expects time between 0 and 1"""
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.pi / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


def apply_func_to_packed_sequence(fn: nn.Module, sequence: PackedSequence) -> PackedSequence:
    return PackedSequence(
        fn(sequence.data), batch_sizes=sequence.batch_sizes, sorted_indices=sequence.sorted_indices,
        unsorted_indices=sequence.unsorted_indices,
    )


class BaseLightningModule(LightningModule):
    def __init__(
        self,
        epochs: int,
        lr: float = 1e-3,
        wd: float = 1e-2,
    ):
        super().__init__()
        self.lr = lr
        self.epochs = epochs
        self.wd = wd
        self.epoch_num = 0

    @abc.abstractmethod
    def get_loss(self, batch, prefix: str) -> torch.Tensor:
        raise NotImplementedError

    def training_step(self, batch, _=None):
        return self.get_loss(batch, "train")

    def validation_step(self, batch, _=None):
        return self.get_loss(batch, "val")

    def test_step(self, batch, _=None):
        return self.get_loss(batch, "test")

    def configure_optimizers(
        self,
    ):
        return configure_optimizer(self.parameters(), lr=self.lr, wd=self.wd, iterations=self.epochs)

    def on_train_epoch_end(self):
        self.epoch_num += 1
        print(f"Epoch: {self.epoch_num}", end="\r")
