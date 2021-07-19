# SYSTEM IMPORTS
from typing import Dict, List, Tuple
from tqdm import tqdm
import argparse
import os
import numpy as np
import torch as pt
import torch.nn.functional as F
import torchvision as ptv


# PYTHON PROJECT


DATASET_ROOT: str = os.path.join("/mnt", "share", "datasets")


class MNISTCNN(pt.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # define the layers of our module
        self.conv1: pt.nn.Module = pt.nn.Conv2d(1, 10, kernel_size=5)
        self.dropout1: pt.nn.Module = pt.nn.Dropout2d()
        self.conv2: pt.nn.Module = pt.nn.Conv2d(10, 20, kernel_size=5)
        self.dropout2: pt.nn.Module = pt.nn.Dropout2d()
        self.fc1: pt.nn.Module = pt.nn.Linear(320, 75)
        self.fc2: pt.nn.Module = pt.nn.Linear(75, 10)
        self.logsoftmax: pt.nn.Module = pt.nn.LogSoftmax(dim=-1)

    def forward(self, X: pt.Tensor) -> pt.Tensor:
        X = F.relu(F.max_pool2d(self.conv1.forward(X), 2))
        X = self.dropout1.forward(X)
        X = F.relu(F.max_pool2d(self.conv2.forward(X), 2))
        X = self.dropout2.forward(X)
        X = X.view(-1, 320)
        X = F.relu(self.fc1.forward(X))
        X = F.relu(self.fc2.forward(X))
        return self.logsoftmax.forward(X)


def train_one_epoch(train_loader: pt.utils.data.DataLoader,
                    m: pt.nn.Module,
                    optimizer: pt.optim.Optimizer,
                    loss_func: pt.nn.Module,
                    gpu_idx: int,
                    epoch: int) -> None:
    m.train()
    pbar_msg: str = "training epoch {0} loss: {1:.6f}"
    current_loss: float = 0
    num_examples_seen: int = 0

    with tqdm(total=len(train_loader),
              desc=pbar_msg.format(epoch, 0)) as pbar:
        for batch_num, (X, Y_gt) in enumerate(train_loader):
            num_examples_seen += X.size(0)

            optimizer.zero_grad()
            Y_hat: pt.Tensor = m.forward(X.cuda(gpu_idx) if gpu_idx >= 0 else X).cpu()
            loss = loss_func.forward(Y_hat, Y_gt)

            current_loss += loss.item()
            loss.backward()
            optimizer.step()

            pbar.set_description(pbar_msg.format(epoch, current_loss/num_examples_seen))
            pbar.update(1)


def eval(test_loader,
         m) -> float:
    m.eval()



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size",
                        type=int,
                        default=2000,
                        help="the batch size")
    parser.add_argument("-p", "--data_path",
                        type=str,
                        default=DATASET_ROOT,
                        help="where to put the data")
    parser.add_argument("-n", "--lr", type=float,
                        default=0.01,
                        help="learning rate")
    parser.add_argument("-m", "--momentum",
                        type=float,
                        default=0,
                        help="momentum")
    parser.add_argument("-e", "--epochs",
                        type=int,
                        default=int(1e3),
                        help="num epochs")
    parser.add_argument("-g", "--gpu",
                        type=int,
                        default=-1,
                        help="gpu idx (-1 is cpu)")
    args = parser.parse_args()

    transform: object = ptv.transforms.Compose([
            ptv.transforms.ToTensor(),
            ptv.transforms.Normalize((0.1307,), (0.3081,))
        ])

    print("loading data")
    train_dataset: object = ptv.datasets.MNIST(
        args.data_path,
        train=True, transform=transform,
        download=True)
    test_dataset: object = ptv.datasets.MNIST(
        args.data_path,
        train=False, transform=transform,
        download=True)

    train_loader: object = pt.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True)
    test_loader: object = pt.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False)

    print("making model")
    m: pt.nn.Module = MNISTCNN().to(args.gpu)
    optimizer = pt.optim.SGD(m.parameters(),
                             lr=args.lr,
                             momentum=args.momentum)
    loss_func: pt.nn.Module = pt.nn.NLLLoss()

    for epoch in range(args.epochs):
        train_one_epoch(train_loader,
                        m,
                        optimizer,
                        loss_func,
                        args.gpu,
                        epoch+1)


if __name__ == "__main__":
    main()