# vim: ts=4 sw=4 et

import os.path
import numpy as np
import time
import torch
from torch import nn
from torch.utils.data import DataLoader

# from bbnn_dataset import BBNNDataset
from mmap_dataset import MemmapDataset

base_dir = 'E:\\extract\\2025\\test-1'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

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
NUM_INPUTS = 30
L1 = 10

def pandas_2_torch(ds):
    return torch.tensor(ds.values.astype(np.float32))

# Define model - the correct one
class BBNNc(nn.Module):
    def __init__(self):
        super().__init__()
        self.input  = nn.Linear(NUM_INPUTS, L1)
        self.output = nn.Linear(2 * L1, 1)

    def forward(self, us, them, w_in, b_in):
        w  = self.input(w_in)
        b  = self.input(b_in)
        l0 = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
        l0 = torch.clamp(l0, 0.0, 1.0)
        x  = self.output(l0)
        return x

# Define model - only test
class BBNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input  = nn.Linear(NUM_INPUTS, L1)
        self.output = nn.Linear(L1, 1)

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

def train(device, dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    start = time.time()
    for batch_no, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y, batch_no)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (batch_no + 1) % 100 == 0:
            tdiff = time.time() - start
            spb = tdiff / (batch_no + 1)
            loss, current = loss.item(), (batch_no + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>7d}/{size:>7d}] {time.strftime('%X %x')}: {spb:>6f} second/batch")

def test(device, dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error Avg loss: {test_loss:>8f} \n")

def main():
    # Training data
    #training_data = BBNNDataset(train_dir,
    #    transform=pandas_2_torch,
    #    target_transform=pandas_2_torch,
    #)
    training_data = MemmapDataset(train_dir)

    # Test data
    #test_data = BBNNDataset(test_dir,
    #    transform=pandas_2_torch,
    #    target_transform=pandas_2_torch,
    #)
    test_data = MemmapDataset(test_dir)

    batch_size = 128

    # Create data loaders.
    train_dataloader = DataLoader(
            training_data,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=3,
            pin_memory=False
        )
    test_dataloader = DataLoader(
            test_data,
            batch_size=batch_size,
            pin_memory=False
        )

    # Just an info:
    for X, y in test_dataloader:
        print(f"Shape of X [N, C]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model = BBNN().to(device)
    print(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(device, train_dataloader, model, loss_fn, optimizer)
        test(device, test_dataloader, model, loss_fn)
    print("Done!")

    save_name = "model2.pth"
    torch.save(model.state_dict(), save_name)
    print(f"Saved PyTorch Model State to {save_name}")

if __name__ == '__main__':
    main()
