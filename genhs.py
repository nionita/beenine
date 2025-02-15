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

# For the model:
NUM_INPUTS = 384
L1 = 16
L2 = 16

debug = False

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
        self.stack = nn.Sequential(
            nn.Linear(NUM_INPUTS * 2, L1),
            nn.Hardsigmoid(),
            nn.Linear(L1, L2),
            nn.ReLU(),
            nn.Linear(L2, 1)
        )

    def forward(self, x):
        pred = self.stack(x)
        return pred.squeeze()

def main_show(args):
    print('Training args:')
    print(args)

    wo = torch.load(args['model'], weights_only=True)

    # We expect the parameters to be weight, bias, weight, bias, ...
    # We do not have the kind of non linearities of the layers
    params = 0
    layers = []
    layer = []
    dim = 768
    exs = 2
    ok = True
    for k in wo.keys():
        if k.endswith('.weight') or k.endswith('.bias'):
            v = wo[k]
            if type(v) == torch.Tensor:
                vs = v.shape
                print(f'{k}: {vs}')
                if len(vs) != exs:
                    print(f'expected shape with {exs} components')
                    ok = False
                    break
                if len(vs) == 1:
                    # We got bias
                    match = vs[0]
                    layer = (ws, np.array(v.tolist()))
                    layers.append(layer)
                    params += vs[0]
                else:   # len(vs) == 2
                    # We got weight
                    match = vs[1]
                    ws = np.array(v.tolist())
                    params += vs[0] * vs[1]
                if match != dim:
                    print(f'expect {dim}, got {match}')
                    ok = False
                    break
                dim = vs[0]
                exs = 3 - exs

    if dim != 1:
        print(f'end dimension is {dim}, expect 1')
        ok = False

    if ok:
        print(f'Number of parameters: {params}')
        for i, (ws, bs) in enumerate(layers):
            print(f'Layer {i}:\nweight {ws.shape} bias {bs.shape}')
        if 'outfile' in args:
            gen_hs(layers, args['outfile'])

def gen_hs(layers, fname):
    with open(fname, 'w') as hf:
        # Generate preambel:
        print('module Eval.Model (', file=hf)
        print('    model', file=hf)
        print(') where', file=hf)
        print(file=hf)
        print('import Eval.NNUE', file=hf)
        print(file=hf)
        # Generate the big accumulators matrix:
        print('acc :: Matrix', file=hf)
        print('acc = makeMatrix [', file=hf)
        mx, ai = layers[0]
        for i in range(mx.shape[1]):
            iprefix = '' if i == 0 else ', '
            print(f'    {iprefix}', end='', file=hf)
            makeAccum(mx[:, i], hf)
        print('    ]', file=hf)
        # Generate the initial accumulator:
        print('aci :: Accum', file=hf)
        print('aci = ', end='', file=hf)
        makeAccum(ai, hf)
        # Generate the intermediate layers:
        layer_names = []
        for i in range(1, len(layers)-1):
            layer_name = f'layer{i}'
            makeLayer(layer_name, layers[i], hf)
            layer_names.append(layer_name)
        # Generate the final layer:
        final_name = 'final_layer'
        makeFinalLayer(final_name, layers[-1], hf)
        # Generate the model:
        print('model :: NNUE', file=hf)
        print('model = makeNNUE acc aci [', end='', file=hf)
        for i, layer_name in enumerate(layer_names):
            prefix = '' if i == 0 else ', '
            print(f'{prefix}{layer_name}', end='', file=hf)
        print(f'] {final_name}', file=hf)

def makeAccum(ac, hf):
    print('makeAccum [', end='', file=hf)
    for j in range(len(ac)):
        jprefix = '' if j == 0 else ', '
        print(f'{jprefix}{ac[j]}', end='', file=hf)
    print(']', file=hf)

def makeLayer(layer_name, layer, hf):
    ws, bs = layer
    print(f'{layer_name} :: Layer', file=hf)
    print(f'{layer_name} = makeLayer (', end='', file=hf)
    print('makeMatrix [', file=hf)
    for i in range(ws.shape[0]):
        iprefix = '' if i == 0 else ', '
        print(f'        {iprefix}', end='', file=hf)
        makeAccum(ws[i, :], hf)
    print('        ])', file=hf)
    print('    (', end='', file=hf)
    makeAccum(bs, hf)
    print('    )', file=hf)

def makeFinalLayer(layer_name, layer, hf):
    ws, bs = layer
    print(f'{layer_name} :: FinalLayer', file=hf)
    print(f'{layer_name} = makeFinalLayer ', end='', file=hf)
    print('(', end='', file=hf)
    makeAccum(ws[0, :], hf)
    print(f'    ) ({bs[0]})', file=hf)

def arg_parser():
    parser = argparse.ArgumentParser(prog='beenine', description='Train BeeNiNe')
    subparsers = parser.add_subparsers(dest='command', help='subcommand help')
    # Show
    parser_show = subparsers.add_parser('show', help='show inference results')
    parser_show.add_argument('-m', '--model', help='model params file')
    parser_show.add_argument('-o', '--outfile', help='generated haskell file name')
    # parser_show.add_argument('-n', '--number', type=int, default=10, help='number inference samples')
    parser_show.set_defaults(func=main_show)
    return parser

if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    args.func(vars(args))
