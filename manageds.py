'''
Count the distribution of a set of positions by the number of different piece types
'''

# vim: ts=4 sw=4 et

import argparse
import glob
import os.path
import sys
#import logging

white_pieces = 'PRNBQ'
black_pieces = 'prnbq'
skip_chars   = 'kK/12345678'
piece_chars  = white_pieces + black_pieces

'''
Process one EPD file and count or sample the combinations
'''
def proc_file(ffn, counts=None, part=None, samples=None, number=0):
    n = 0
    with open(ffn, 'r') as f:
        for line in f:
            line = line[:-1]
            n += 1
            #print(line)
            me, yo = proc_line(line)
            #print(f'{me} - {yo}')
            if me:
                code = f'{me}-{yo}'
                if counts is not None:
                    if code in counts:
                        counts[code] += 1
                    else:
                        counts[code] = 1
                if samples is not None and part is not None:
                    if me == part or yo == part:
                        samples.append((code, line))
                        if number > 0 and len(samples) >= number:
                            break
    return n

'''
Process one EPD line and return the codes for the moving and non-moving parts
'''
def proc_line(line):
    pos = { 'P': 0, 'p': 0, 'R': 0, 'r': 0, 'n': 0, 'N': 0, 'b': 0, 'B': 0, 'q': 0, 'Q': 0 }
    spaces = 0
    moving = None
    for i in range(len(line)):
        c = line[i]
        if spaces == 1 and (c == 'w' or c == 'b'):
            moving = c
        elif c in piece_chars:
            pos[c] += 1
        elif c == ' ':
            if spaces == 0:
                spaces = 1
            else:
                break
    code_white = f'{pos["P"]}{pos["R"]}{pos["N"]}{pos["B"]}{pos["Q"]}'
    code_black = f'{pos["p"]}{pos["r"]}{pos["n"]}{pos["b"]}{pos["q"]}'
    if moving == 'w':
        return code_white, code_black
    elif moving == 'b':
        return code_black, code_white
    else:
        return '', ''

'''
Statistics over all files
'''
def do_stats(args):
    print(f'Statistics over files in {args.dir}:')
    counts = {}
    total = 0
    for fn in glob.glob('*.epd', root_dir=args.dir):
        ffn = os.path.join(args.dir, fn)
        lines = proc_file(ffn, counts)
        print(f'{ffn}: {lines} lines')
        total += lines

    with open(args.out, 'w') as f:
        for code in sorted(counts.keys()):
            print(f'{code}: {counts[code]}', file=f)
    print(f'Total: {total} positions with {len(counts.keys())} keys')

'''
Samples over all files
'''
def do_sample(args):
    print(f'Samples from files in {args.dir}:')
    samples = []
    for fn in glob.glob('*.epd', root_dir=args.dir):
        ffn = os.path.join(args.dir, fn)
        lines = proc_file(ffn, part=args.side, samples=samples, number=args.number)
        if args.number > 0 and len(samples) >= args.number:
            break

    with open(args.out, 'w') as f:
        for code, fen in sorted(samples):
            print(f'{code}: {fen}', file=f)

# Parse command line arguments
parser = argparse.ArgumentParser(
            prog='distib',
            description='Count position distribution in epd files by piece types')
#parser.add_argument('--debug', action='store_true', help='enable debug')
subparsers = parser.add_subparsers(title='subcommands', required=True)
parser_stats = subparsers.add_parser('stats')
parser_stats.add_argument('-d', '--dir', required=True, help='directory with EPD files')
parser_stats.add_argument('-o', '--out', default=sys.stdout, help='output file name')
parser_stats.set_defaults(func=do_stats)
parser_sample = subparsers.add_parser('sample')
parser_sample.add_argument('-d', '--dir', required=True, help='directory with EPD files')
parser_sample.add_argument('-o', '--out', default=sys.stdout, help='output file name')
parser_sample.add_argument('-s', '--side', required=True, help='side sample to search for')
parser_sample.add_argument('-n', '--number', type=int, default=0, help='number of samples (default: all)')
parser_sample.set_defaults(func=do_sample)
args = parser.parse_args()
args.func(args)
