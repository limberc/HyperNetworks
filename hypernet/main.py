from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import torch
import torch.nn.parallel
import torch.optim as optim
from pytorch_lightning.core import LightningModule
from pytorch_lightning.loggers.neptune import NeptuneLogger
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from resnet import resnet18


def get_dataset(data_path, dataset):
    if dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32, padding=4),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])
    if dataset == 'cifar10':
        train_data = CIFAR10(data_path, train=True, transform=train_transform, download=True)
        test_data = CIFAR10(data_path, train=False, transform=test_transform, download=True)
    elif dataset == 'cifar100':
        train_data = CIFAR100(data_path, train=True, transform=train_transform, download=True)
        test_data = CIFAR100(data_path, train=False, transform=test_transform, download=True)
    return train_data, test_data


class HyperResNetCIFAR(LightningModule):
    """
    1. refactor to higher and Transformers structure.
    """

    def __init__(
            self,
            lr: float,
            momentum: float,
            weight_decay: int,
            data_path: str,
            batch_size: int,
            workers: int,
            dataset: str,
            base: int,
            z_dim: int,
            **kwargs, ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.workers = workers
        self.model = resnet18(num_classes=100 if dataset == 'cifar100' else 10,
                              base=base, z_dim=z_dim)
        self.train_datset, self.test_dataset = get_dataset(data_path, dataset)
        self.steps_per_epoch = len(self.train_dataloader())

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss_func = nn.CrossEntropyLoss()
        loss_val = loss_func(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log('train_loss', loss_val, on_epoch=True, logger=True)
        self.log('train_acc1', acc1, prog_bar=True, on_epoch=True, logger=True)
        self.log('train_acc5', acc5, on_epoch=True, logger=True)
        return loss_val

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            dataset=self.train_datset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True
        )
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True
        )
        return val_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True
        )
        return test_loader

    def validation_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss_func = nn.CrossEntropyLoss()
        loss_val = loss_func(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log('val_loss', loss_val, on_step=True, on_epoch=True)
        self.log('val_acc1', acc1, on_step=True, prog_bar=True, on_epoch=True)
        self.log('val_acc5', acc5, on_step=True, on_epoch=True)
        return loss_val

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    @staticmethod
    def __accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=(self.lr or self.learning_rate),
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=True
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[60, 120, 160], gamma=0.2
        )
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                            help='number of data loading workers (default: 4)')
        parser.add_argument('-b', '--batch-size', default=512, type=int,
                            metavar='N',
                            help='mini-batch size (default: 512), this is the total '
                                 'batch size of all GPUs on the current node when '
                                 'using Data Parallel or Distributed Data Parallel')
        parser.add_argument('-lr', '--learning-rate', default=0.1, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                            metavar='W', help='weight decay (default: 5e-4)',
                            dest='weight_decay')
        parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                            help='use pre-trained model')
        parser.add_argument('--base', default=32, type=int,
                            help='base for HyperNet.')
        parser.add_argument('--z_dim', default=512, type=int,
                            help='z_dim for HyperNet.')
        return parser


def main(args: Namespace) -> None:
    if args.seed is not None:
        pl.seed_everything(args.seed)

    if args.distributed_backend == 'ddp':
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.batch_size = int(args.batch_size / max(1, args.gpus))
        args.workers = int(args.workers / max(1, args.gpus))

    args.logger = NeptuneLogger(
        project_name='YOUR_PROJ/hypermixup',
        experiment_name="experiment_name",
        params={
            'model': "ResNet18",
            'hypernet': True,
            'dataset': args.dataset,
            'base': args.base,
            'z_dim': args.z_dim,
            'learning_rate': args.lr,
        }
    )

    model = HyperResNetCIFAR(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args)

    if args.evaluate:
        trainer.test(model)
    else:
        trainer.fit(model)


def run_cli():
    parent_parser = ArgumentParser(add_help=False)
    # ------------
    # args
    # ------------
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument('--data-path', metavar='DIR', type=str, default='./data',
                               help='path to dataset')
    parent_parser.add_argument('--dataset', metavar='DIR', type=str, default='cifar100',
                               choices=('cifar10', 'cifar100'), help='CIFAR10 or CIFAR100.')
    parent_parser.add_argument('--logname', type=str, default='baseline',
                               help='path to dataset')
    parent_parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                               help='evaluate model on validation set')
    parent_parser.add_argument('--seed', type=int, default=42,
                               help='seed for initializing training.')

    parser = HyperResNetCIFAR.add_model_specific_args(parent_parser)
    parser.set_defaults(
        profiler=True,
        deterministic=True,
        max_epochs=200,
        accelerator='ddp',
        plugins='ddp_sharded'
    )
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run_cli()
