# vim: ts=4 sw=4 et

import argparse
import os.path
import numpy as np
import time
import torch
import sys
from torch import nn
from torch.utils.data import DataLoader

from bbnn_dataset import BBNNDataset
#from mmap_dataset import MemmapDataset

base_dir  = 'E:\\extract\\2025\\test-1'
train_dir = os.path.join(base_dir, 'train')
test_dir  = os.path.join(base_dir, 'test')

train_pos = 10_423_404
test_pos  =  1_000_000

# Workers for the datasets
WORKERS = None

# Why do we need to scale the network score prediction? It seems that
# this is necessary in order to keep weigths & biases small,
# so we chose a scale of 1000 - see PRED_SCALE
#
# Then we need to stretch the sigmoid by the centipawn score, like:
# for how big a score are we almost winning (sigmoid approaches to 1)?
# Now: sigmoid(4) = 0.982
# We want to have that win probability for a score of 600 cp
# Then the stretch factor must be 1 / 150
PRED_SCALE = 1000.0
SCORE_SIGMOID_SCALE = 1.0 / 150.0
PRED_SIGMOID_SCALE = PRED_SCALE * SCORE_SIGMOID_SCALE

# For the model:
NUM_INPUTS = 384
L1 = 128
L2 = 128

# Define model - the correct one
class BBNNc(nn.Module):
    def __init__(self):
        super().__init__()
        self.side = nn.Linear(NUM_INPUTS, L1)
        self.inte = nn.Linear(2 * L1, L2)
        self.outp = nn.Linear(L2, 1, bias=False)

    def forward(self, x):
        # Active / passive side input representation
        a_in, p_in = torch.tensor_split(x, 2, dim=1)
        a   = self.side(a_in)
        p   = self.side(p_in)
        c   = torch.cat([a, p], dim=1)
        # l0a = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
        l0  = torch.clamp(c, 0.0, 1.0)
        i   = self.inte(l0)
        l1  = torch.clamp(i, 0.0, 1.0)
        y   = self.outp(l1)
        return y

# Define model - only test
class BBNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input  = nn.Linear(NUM_INPUTS * 2, L1)
        self.output = nn.Linear(L1, 1, bias=False)

    def forward(self, x):
        w  = self.input(x)
        l0 = torch.clamp(w, 0.0, 1.0)
        y  = self.output(l0)
        return y

# Here: we don't have us, them in the features!
def loss_fn(pred, y, batch_no = 0):
    #score, outcome = y

    wdl_eval_model  = (pred * PRED_SIGMOID_SCALE).sigmoid()
    wdl_eval_target = (y    * SCORE_SIGMOID_SCALE).sigmoid()

    mloss = torch.abs(wdl_eval_target - wdl_eval_model).square().mean()
    #if (batch_no + 1) % 100 == 0:
    #    print(f'wdl_eval: model = {wdl_eval_model} target = {wdl_eval_target}')

    return mloss

def train(device, dataloader, model, loss_fn, optimizer, num_batches=1000):
    print(f'Begin train on {device} with {num_batches}')
    #size = len(dataloader.dataset)
    start = time.time()
    model.train()
    batch_no = 0
    for X, y in dataloader:
        batch_no += 1
        X, y = X.to(device), y.to(device)
        # print(f'Batch {batch_no}: {X} -> {y}')

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y, batch_no)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_no % 1000 == 0:
            tdiff = time.time() - start
            spb = tdiff / (batch_no + 1)
            nows = time.strftime('%X %x')
            loss, current = loss.item(), batch_no * len(X)
            size = num_batches * len(X)
            print(f"loss: {loss:>7f} [{current:>7d}/{size:>7d}] {nows}: {spb:>6f} seconds/batch")
        if batch_no >= num_batches:
            return

def test(device, dataloader, model, loss_fn, num_batches=10):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        batch_no = 0
        for X, y in dataloader:
            batch_no += 1
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            if batch_no >= num_batches:
                break
    test_loss /= num_batches
    print(f"Test Error Avg loss: {test_loss:>8f} \n")
    return test_loss

def main_train(args):
    # Training data
    training_data = BBNNDataset(train_dir)

    # Test data
    test_data = BBNNDataset(test_dir)

    batch_size = args['batch']

    # Create data loaders.
    train_dataloader = DataLoader(
            training_data,
            batch_size=batch_size,
            pin_memory=False
        )
    test_dataloader = DataLoader(
            test_data,
            batch_size=batch_size,
            pin_memory=False
        )

    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model = BBNNc()
    if 'restore' in args and args['restore'] is not None:
        print(f'Restore model weights from {args["restore"]}')
        model.load_state_dict(torch.load(args['restore'], weights_only=True))

    model = model.to(device)
    print(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=args['rate'], momentum=args['momentum'])

    epochs = args['epochs']
    num_batches_train = int(train_pos / batch_size)
    num_batches_test  = int(test_pos / batch_size)

    test_losses = []

    # First evaluation: completely random - for comparison
    test_loss = test(device, test_dataloader, model, loss_fn, num_batches=num_batches_test)
    test_losses.append(test_loss)

    for t in range(epochs):
        print(f"Epoch {t+1} from {epochs}\n-------------------------------")
        train(device, train_dataloader, model, loss_fn, optimizer, num_batches=num_batches_train)
        save_name = f"{args['save']}-{t}.pth"
        torch.save(model.state_dict(), save_name)
        print(f"Saved PyTorch Model State to {save_name}")
        test_loss = test(device, test_dataloader, model, loss_fn, num_batches=num_batches_test)
        test_losses.append(test_loss)

    print(f"Test losses: {test_losses}")
    print("Done!")

def arg_parser():
    parser = argparse.ArgumentParser(prog='beenine', description='Train BeeNiNe')
    subparsers = parser.add_subparsers(dest='command', help='subcommand help')
    parser_train = subparsers.add_parser('train', help='train the network')
    parser_train.add_argument('-e', '--epochs', type=int, default=10, help='epochs to train')
    parser_train.add_argument('-l', '--rate', type=float, default=0.001, help='learning rate')
    parser_train.add_argument('-m', '--momentum', type=float, default=0, help='momentum for SGD')
    parser_train.add_argument('-b', '--batch', type=int, default=256, help='bach size')
    parser_train.add_argument('-r', '--restore', help='restore model params from file')
    parser_train.add_argument('-s', '--save', default='model', help='save model params to file')
    parser_train.set_defaults(func=main_train)
    return parser

if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    args.func(vars(args))
