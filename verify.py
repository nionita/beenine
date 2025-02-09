# vim: ts=4 sw=4 et

import numpy as np
import chess

# The index coding is line this:
# the indexes between 0 and 383 are for the part to move, from 384 to 767 for the passive one
# Inside one part there are 64 squares per piece type: pawn, knight, bishop, rook, queen and king
# So indexes between 0 and 63 are pawns of the part to move, from 64 to 127 are knights
# of the part to move, then bishops and so on
# Indexes between 384 and 447 are pawns of the passive part, from 448 to 511 are knights
# of the passive part, then bishops and so on
# The squares are not absolute, but from the point of view of one part, by miroring (2 possibilities):
# - simple: pov of the piece part
# - direct: pov of the moving part

direct = True

'''
Given an index, return piece and square
'''
def decode_index(idx, moving):
    pc_type = idx // 64 + 1   # chess piece type coding begins at 1
    assert pc_type <= 12
    pc_square = idx % 64

    # Piece type and color
    color = moving
    if pc_type > 6:
        color = not color
        pc_type -= 6

    # Mirror the square depending on part or moving part
    pov = moving if direct else color
    if pov == chess.BLACK:
        pc_square = chess.square_mirror(pc_square)

    return chess.Piece(pc_type, color), pc_square

def board_from_indices(idxs, color):
    piece_map = {}
    for idx in idxs:
        piece, square = decode_index(idx, color)
        piece_map[square] = piece
    board = chess.BaseBoard.empty()
    board.set_piece_map(piece_map)
    return board

def main():
    in_filt = r'E:\extract\2025\test-1\train\s49m-ak-filt.csv'
    in_feat = r'E:\extract\2025\test-1\train\s49m-ak-feat.txt'
    test = -1

    good = 0
    bad = 0

    with open(in_filt, 'r') as filt:
        with open(in_feat, 'r') as feat:
            i = 0
            while test < 0 or i < test:
                try:
                    csv_line = next(filt)
                    txt_line = next(feat)
                except StopIteration:
                    break

                i += 1
                ifen, color, _ = csv_line.split(' ', maxsplit=2)
                idxs = np.fromstring(txt_line, dtype=int, sep=',')

                if color == 'w':
                    color = chess.WHITE
                else:
                    color = chess.BLACK

                board = board_from_indices(idxs, color)
                fen = board.board_fen()

                if fen == ifen:
                    good += 1
                else:
                    bad += 1
                    print(f'csv: {ifen} + {color}')
                    print(f'txt: {txt_line.strip()} -> {idxs}')
                    print(f'boa: {fen}')
                    print('----------------')

    print(f'records total/good/bad: {i} / {good} / {bad}')

if __name__ == '__main__':
    main()
