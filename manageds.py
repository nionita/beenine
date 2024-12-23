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
Process one EPD line and return the tuples for the moving and passive parts (P, R, N, B, Q)
'''
def one_line(line):
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
    tuple_white = (pos["P"], pos["R"], pos["N"], pos["B"], pos["Q"])
    tuple_black = (pos["p"], pos["r"], pos["n"], pos["b"], pos["q"])
    if moving == 'w':
        return tuple_white, tuple_black
    elif moving == 'b':
        return tuple_black, tuple_white
    else:
        return (), ()

'''
Generator delivers one line at a time with further information, all in a dictionary:
record number, file number, file name, line number, line, tuple moving part, tuple passive part
'''
def data_generator(root_dir, glob_pat='*.epd'):
    rec_no  = 0
    file_no = 0
    for fn in glob.glob(glob_pat, root_dir=root_dir):
        file_no += 1
        ffn = os.path.join(root_dir, fn)
        with open(ffn, 'r') as f:
            line_no = 0
            for line in f:
                rec_no += 1
                line_no += 1
                line = line[:-1]
                tmp, tpp = one_line(line)
                yield { 'rec_no': rec_no, 'file_no': file_no, 'file_name': ffn,
                        'line_no': line_no, 'line': line, 'moving_part': tmp, 'passive_part': tpp }

'''
Make a code from the tuple
'''
def tuple_to_code(tup):
    if len(tup) != 5:
        return 'XXXXX'
    else:
        p, r, n, b, q = tup
        return f'{p}{r}{n}{b}{q}'

'''
Statistics over all files
'''
def do_stats(args):
    print(f'Statistics over files in {args.dir}:')
    counts = {}
    for line_info in data_generator(root_dir=args.dir):
        mp = tuple_to_code(line_info['moving_part'])
        pp = tuple_to_code(line_info['passive_part'])
        code = f'{mp}-{pp}'
        if code in counts:
            counts[code] += 1
        else:
            counts[code] = 1

        # Report progress
        rn = line_info['rec_no']
        if args.progress > 0 and rn % args.progress == 0:
            fn = line_info['file_name']
            ln = line_info['line_no']
            print(f'Record {rn}, file {fn}, line {ln}')

        # Limit the input
        if args.limit > 0 and rn >= args.limit:
            break

    with open(args.out, 'w') as f:
        for code in sorted(counts.keys()):
            print(f'{code}: {counts[code]}', file=f)
    print(f'Total: {rn} positions with {len(counts.keys())} keys')

'''
Samples over all files
'''
def do_sample(args):
    print(f'Samples from files in {args.dir}:')
    samples = []
    for line_info in data_generator(root_dir=args.dir):
        mp = tuple_to_code(line_info['moving_part'])
        pp = tuple_to_code(line_info['passive_part'])
        if mp == args.side or pp == args.side:
            code = f'{mp}-{pp}'
            samples.append((code, line_info['line']))
            if args.number > 0 and len(samples) >= args.number:
                break

        # Report progress
        rn = line_info['rec_no']
        if args.progress > 0 and rn % args.progress == 0:
            fn = line_info['file_name']
            ln = line_info['line_no']
            print(f'Record {rn}, file {fn}, line {ln}')

        # Limit the input
        if args.limit > 0 and rn >= args.limit:
            break

    with open(args.out, 'w') as f:
        for code, fen in sorted(samples):
            print(f'{code}: {fen}', file=f)

# Parse command line arguments
parser = argparse.ArgumentParser(
            prog='manageds',
            description='Count / sample / filter EPD lines in epd files by piece types')
#parser.add_argument('--debug', action='store_true', help='enable debug')
parser.add_argument('-d', '--dir', required=True, help='directory with EPD files')
parser.add_argument('-o', '--out', default=sys.stdout, help='output file name')
parser.add_argument('-p', '--progress', type=int, default=0, help='report progress')
parser.add_argument('-l', '--limit', type=int, default=0, help='limit input')
subparsers = parser.add_subparsers(title='subcommands', required=True)
parser_stats = subparsers.add_parser('stats')
parser_stats.set_defaults(func=do_stats)
parser_sample = subparsers.add_parser('sample')
parser_sample.add_argument('-s', '--side', required=True, help='side sample to search for')
parser_sample.add_argument('-n', '--number', type=int, default=0, help='number of samples (default: all)')
parser_sample.set_defaults(func=do_sample)
args = parser.parse_args()
args.func(args)
