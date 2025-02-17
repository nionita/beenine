# vim: ts=4 sw=4 et

# Read a model from a pth file and output it in JSON format
# which can be transformed by JsonToBin (a utility from Barbarossa)
# to a binery format understood by Barbarossa

# The call would then be:
#
# python jsonmodel.py -m model.pth | JsonToBin -o model.bin
#

import configparser
import argparse
import os.path
import numpy as np
import time
import torch
import sys
import json
from torch import nn
from torch.utils.data import DataLoader

debug = False

def main(args):
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
            gen_json(layers, args['activation'], args['outfile'])

def gen_json(layers, activation, fname):
    with open(fname, 'w') as hf:
        # We construct the python object and then jsonify
        # Accum is a list of floats,
        # Matrix is a list of Accum,
        # and the rest are objects
        nnue = {}
        # Generate the big accumulators matrix:
        nnueAccums = []
        mx, ai = layers[0]
        for i in range(mx.shape[1]):
            accum = list(map(float, mx[:, i]))
            nnueAccums.append(accum)
        nnue['nnueAccums'] = nnueAccums
        nnue['nnueBias'] = list(map(float, ai))
        nnue['nnueNonlin'] = activation
        # Generate the intermediate layers:
        mlayers = []
        for i in range(1, len(layers)-1):
            mlayer = {}
            laWeights = []
            ws, bs = layers[i]
            for i in range(ws.shape[0]):
                laWeights.append(list(map(float, ws[i, :])))
            mlayer['laWeights'] = laWeights
            mlayer['laBias'] = list(map(float, bs))
            mlayer['laNonlin'] = activation
            mlayers.append(mlayer)
        nnue['nnueLayers'] = mlayers
        # Generate the final layer:
        final_layer = {}
        ws, bs = layers[-1]
        final_layer['flWeights'] = list(map(float, ws[0, :]))
        final_layer['flBias'] = float(bs[0])
        nnue['nnueFinal'] = final_layer
        # Output the jsonified model:
        print(json.dumps(nnue, indent=2), file=hf)
        # print(nnue)

def arg_parser():
    parser = argparse.ArgumentParser(prog='jsonmodel', description='Jsonify a pth file')
    parser.add_argument('-m', '--model', help='model params file')
    parser.add_argument('-o', '--outfile', help='output file')
    parser.add_argument('-a', '--activation', default="ReLU", help='activation function')
    parser.set_defaults(func=main)
    return parser

if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    args.func(vars(args))
