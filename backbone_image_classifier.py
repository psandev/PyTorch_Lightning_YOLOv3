# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import ArgumentParser
from wcmatch.pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pl_examples import _DATASETS_PATH, _TORCHVISION_AVAILABLE, _TORCHVISION_MNIST_AVAILABLE, cli_lightning_logo
from pytorch_lightning.callbacks import GPUStatsMonitor,ModelCheckpoint
if _TORCHVISION_AVAILABLE:
    from torchvision import transforms
if _TORCHVISION_MNIST_AVAILABLE:
    from torchvision.datasets import MNIST
else:
    from tests.helpers.datasets import MNIST


class Backbone(torch.nn.Module):
    """
    >>> Backbone()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Backbone(
      (l1): Linear(...)
      (l2): Linear(...)
    )
    """

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x


class LitClassifier(pl.LightningModule):
    """
    >>> LitClassifier(Backbone())  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    LitClassifier(
      (backbone): ...
    )
    """

    def __init__(self, backbone, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = backbone

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.backbone(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitClassifier")
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parent_parser


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitClassifier.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dataset = MNIST(_DATASETS_PATH, train=True, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST(_DATASETS_PATH, train=False, download=True, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    train_loader = DataLoader(mnist_train, batch_size=args.batch_size)
    val_loader = DataLoader(mnist_val, batch_size=args.batch_size)
    test_loader = DataLoader(mnist_test, batch_size=args.batch_size)

    # ------------
    # model
    # ------------
    model = LitClassifier(Backbone(hidden_dim=args.hidden_dim), args.learning_rate)

    # ------------
    # training
    # ------------
    experiment_dir = Path.cwd()
    checkpoint_callback = ModelCheckpoint(dirpath=experiment_dir)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader,gradient_clip_val=0.5,
                         benchmark=True,
                         callbacks=[GPUStatsMonitor(),checkpoint_callback],)

    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=test_loader, gradient_clip_val=0.5,
                         benchmark=True,
                         callbacks=[GPUStatsMonitor(),checkpoint_callback],)
    print(result)


if __name__ == '__main__':
    cli_lightning_logo()
    cli_main()
