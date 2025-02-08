# vim: ts=4 sw=4 et

import configparser
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

# Workers for the datasets
WORKERS = None

# Why do we need to scale the network score prediction? It seems that
# this is necessary in order to keep weigths & biases small,
# so we chose a scale of 1000 - see PRED_SCALE
#
# For the loss function we compare the sigmoid of the scores (pred vs target)
# in order to focus more on smaller absolute scores
# Then we need to stretch the sigmoid by the centipawn score, like:
# for how big a score are we almost winning (sigmoid approaches to 1)?
# Now: sigmoid(4) = 0.982
# We want to have that win probability for a score of 600 cp
# Then the stretch factor must be 1 / 150
PRED_SCALE = 1000.0
SCORE_SIGMOID_SCALE = 1.0 / 150.0

# For the model:
NUM_INPUTS = 384
L1 = 32
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
        return y * PRED_SCALE

# Define model - only test
class BBNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input  = nn.Linear(NUM_INPUTS * 2, L1, bias=False)
        self.output = nn.Linear(L1, 1, bias=False)

    def forward(self, x):
        l0  = self.input(x)
        l0n = torch.clamp(l0, 0.0, 1.0)
        y   = self.output(l0n) * PRED_SCALE
        return y

# Here: we don't have us, them in the features!
def loss_fn(pred, y, batch_no = 0):
    #score, outcome = y

    wdl_eval_model  = (pred * SCORE_SIGMOID_SCALE).sigmoid()
    wdl_eval_target = (y    * SCORE_SIGMOID_SCALE).sigmoid()

    mloss = torch.abs(wdl_eval_target - wdl_eval_model).square().mean()
    #if (batch_no + 1) % 100 == 0:
    #    print(f'wdl_eval: model = {wdl_eval_model} target = {wdl_eval_target}')

    return mloss

def train(device, dataloader, model, loss_fn, optimizer, train_pos):
    print(f'Train on {device} with {train_pos} positions')
    start = time.time()
    train_inst = 0
    train_loss = 0
    batch_report = None
    model.train()
    batch_no = 0
    for X, y in dataloader:
        batch_no += 1
        n = X.shape[0]
        train_inst += n
        if batch_report is None:
            batch_report = int(200000 / n)
        X, y = X.to(device), y.to(device)
        # print(f'Batch {batch_no}: {X} -> {y}')

        optimizer.zero_grad()
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y, batch_no)

        # Backpropagation
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * n

        if batch_no % batch_report == 0:
            tdiff = time.time() - start
            ips = round(train_inst / tdiff)
            nows = time.strftime('%X %x')
            mloss = train_loss / train_inst
            print(f"loss: {mloss:>7f} [{train_inst:>7d}/{train_pos:>7d}] {nows}: {ips:>6d} samples/second")

    nows = time.strftime('%X %x')
    mloss = train_loss / train_inst
    print(f"Epoch loss: {mloss:>7f} [{train_inst:>7d}/{train_inst:>7d}] {nows}")
    return train_inst, mloss

def test(device, dataloader, model, loss_fn):
    model.eval()
    test_inst = 0
    test_loss = 0
    with torch.no_grad():
        batch_no = 0
        for X, y in dataloader:
            batch_no += 1
            n = X.shape[0]
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item() * n
            test_inst += n

    test_loss /= test_inst
    print(f"Test Error Avg loss: {test_loss:>8f} \n")
    return test_loss

def evaluate(device, dataloader, model, num):
    model.eval()
    eval_inst = 0
    with torch.no_grad():
        batch_no = 0
        for X in dataloader:
            batch_no += 1
            n = X.shape[0]
            X = X.to(device)
            pred = model(X)
            print(f'Eval instances {eval_inst + 1} to {eval_inst + n}: {pred}')
            eval_inst += n
            if eval_inst >= num:
                return

def main_train(args):
    # Training data
    training_data = BBNNDataset(args['train_dir'])

    # Test data
    test_data = BBNNDataset(args['test_dir'])

    batch_size = args['batch']

    # Create data loaders.
    train_dataloader = DataLoader(
            training_data,
            batch_size=batch_size,
            num_workers=args['workers'],
            pin_memory=(args['workers'] > 1)
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

    # model = BBNNc()
    model = BBNN()
    if 'restore' in args and args['restore'] is not None:
        print(f'Restore model weights from {args["restore"]}')
        model.load_state_dict(torch.load(args['restore'], weights_only=True))

    model = model.to(device)
    print(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=args['rate'], momentum=args['momentum'])
    #optimizer = torch.optim.AdamW(model.parameters(), lr=args['rate'])
    # optimizer = torch.optim.AdamW(model.parameters())

    epochs = args['epochs']

    train_losses = []
    test_losses = []

    # First evaluation: completely random - for comparison
    test_loss = test(device, test_dataloader, model, loss_fn)
    test_losses.append(test_loss)

    train_pos = 0
    start = time.time()

    for t in range(epochs):
        print(f"Epoch {t+1} from {epochs}\n-------------------------------")
        train_pos, train_loss = train(device, train_dataloader, model, loss_fn, optimizer, train_pos)
        train_losses.append(train_loss)
        save_name = f"{args['save']}-{t}.pth"
        torch.save(model.state_dict(), save_name)
        print(f"Saved PyTorch Model State to {save_name}")
        test_loss = test(device, test_dataloader, model, loss_fn)
        test_losses.append(test_loss)

        tdiff = time.time() - start
        spe = tdiff / (t + 1)
        rem = round((epochs - t - 1) * spe)
        if t + 1 < epochs:
            spe = round(spe)
            print(f"{spe} seconds per epoch - {rem} seconds remaining\n-------------------------------")

    print(f"Done after {tdiff} seconds")
    print(f"Train/test losses:")
    for i in range(len(test_losses)):
        if i == 0:
            trl = "        "
        else:
            trl = f"{train_losses[i-1]:>7f}"
        print(f"{trl} --> {test_losses[i]:>7f}")

def main_show(args):
    # Inference data
    feature_data = BBNNDataset(args['feature_file'], evaluate=True, skip=args['skip'])

    batch_size = min(args['batch'], args['number'])

    # Create data loaders.
    feature_dataloader = DataLoader(
            feature_data,
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

    # model = BBNNc()
    model = BBNN()
    if 'model' in args and args['model'] is not None:
        print(f'Evaluate model {args["model"]}')
        model.load_state_dict(torch.load(args['model'], weights_only=True))

    model = model.to(device)
    print(model)

    # Evaluation:
    evaluate(device, feature_dataloader, model, args['number'])

def arg_parser(config):
    parser = argparse.ArgumentParser(prog='beenine', description='Train BeeNiNe')
    subparsers = parser.add_subparsers(dest='command', help='subcommand help')
    # Train
    parser_train = subparsers.add_parser('train', help='train the network')
    parser_train.add_argument('-e', '--epochs', type=int,
            default=config.getint('DEFAULT', 'epochs', fallback=10),
            help='epochs to train')
    parser_train.add_argument('-l', '--rate', type=float,
            default=config.getfloat('DEFAULT', 'rate', fallback=0.001),
            help='learning rate')
    parser_train.add_argument('-m', '--momentum', type=float,
            default=config.getint('DEFAULT', 'momentum', fallback=0),
            help='momentum for SGD')
    parser_train.add_argument('-b', '--batch', type=int,
            default=config.getint('DEFAULT', 'batch', fallback=256),
            help='bach size')
    parser_train.add_argument('-w', '--workers', type=int,
            default=config.getint('DEFAULT', 'workers', fallback=1),
            help='number of dataset workers')
    parser_train.add_argument('-r', '--restore', help='restore model params from file')
    parser_train.add_argument('-s', '--save', default='model', help='save model params to file')
    parser_train.add_argument('-t', '--train_dir',
            default=config.get('DEFAULT', 'train_dir', fallback='train'),
            help='directory with training data')
    parser_train.add_argument('-v', '--test_dir',
            default=config.get('DEFAULT', 'test_dir', fallback='test'),
            help='directory with test data')
    parser_train.set_defaults(func=main_train)
    # Show
    parser_show = subparsers.add_parser('show', help='show inference results')
    parser_show.add_argument('-b', '--batch', type=int,
            default=config.getint('DEFAULT', 'batch', fallback=256),
            help='bach size')
    parser_show.add_argument('-m', '--model', help='model params file')
    parser_show.add_argument('-f', '--feature_file', help='file with features data')
    parser_show.add_argument('-s', '--skip', type=int, default=0, help='skip samples')
    parser_show.add_argument('-n', '--number', type=int, default=10, help='number inference samples')
    parser_show.set_defaults(func=main_show)
    return parser

def config_defaults():
    base_dir  = os.path.join('E:', r'\extract', '2025', 'test-1')
    train_dir = os.path.join(base_dir, 'train')
    test_dir  = os.path.join(base_dir, 'test')

    cds = {
            'DEFAULT': {
                'epochs': 10,
                'rate': 0.001,
                'momentum': 0,
                'batch': 256,
                'workers': 1,
                'train_dir': train_dir,
                'test_dir': test_dir,
                }
        }
    return cds

def config_parser():
    config = configparser.ConfigParser()
    config.read_dict(config_defaults())
    config.read(['config.ini'])
    return config

if __name__ == '__main__':
    config = config_parser()
    parser = arg_parser(config)
    args = parser.parse_args()
    args.func(vars(args))
