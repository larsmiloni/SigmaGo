from pysgf import SGF, SGFNode
from typing import List
import numpy as np
from goEnv import GoGame

root = SGF.parse_file(
    'data/Top50/nine/nine_strong/2014-12-30T12:35:43.567Z_mc4tw1rrez6z.sgf')

x_dict = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7,
    'i': 8
}

y_dict = {
    'a': 8,
    'b': 7,
    'c': 6,
    'd': 5,
    'e': 4,
    'f': 3,
    'g': 2,
    'h': 1,
    'i': 0
}


def remove_deadstones(b_moves: np.ndarray, w_moves: np.ndarray, move: str) -> List[np.ndarray]:
    board = b_moves + (w_moves * 2)
    game = GoGame(board=board)
    game.turn = 'B' if move[2] == 'B' else 'W'
    game.check_captures(move[0], move[1])


"""Get the invalid moves """


def invalid_moves(b_moves: np.ndarray, w_moves: np.ndarray) -> np.ndarray:
    board = b_moves + (w_moves * 2)
    game = GoGame(board=board)
    game.turn = 'B' if move[2] == 'B' else 'W'
    # Metoden er avhening av history variabelen i GoGame
    valid_moves = game.get_legal_actions()
    valid_moves.remove('pass')
    valid_moves.remove('resign')

    valid_moves_array = np.zeros((9, 9))
    for move in valid_moves:
        i, j = move
        # i = move[1]
        # j = move[3]
        valid_moves_array[i][j] = 1

    board_size = game.size
    return np.ones((board_size, board_size)) - valid_moves_array


def sgf_to_array(move: str) -> np.ndarray:
    coords = np.zeros([9, 9])
    coords[x_dict.get(move[0])][y_dict.get(move[1])] = 1
    return coords


def get_states(root: SGFNode) -> List[np.ndarray]:
    """Retrieve all SGF coordinate values for moves in a linear SGF game without variations."""
    b_pieces = []
    w_pieces = []
    turns = []
    invalid_moves = []
    passes = []
    is_over = []
    board = []
    current_node = root
    board_size = current_node.board_size  # Use the board size from the root node
    while current_node:
        move = current_node.move
        if move:
            invalid_moves.append(invalid_moves(b_pieces, w_pieces))
            parent = current_node.parent
            if current_node.children[0]:
                is_over.append(np.zeros((9, 9)))
            else:
                """If it's a leaf node, it's the last move"""
                is_over.append(np.ones(9, 9))

            if parent.move and parent.move.is_pass:
                """Previous move was a pass"""
                passes.append(np.ones((9, 9)))
            else:
                passes.append(np.zeros((9, 9)))

            if current_node.player == 'B':
                turns.append(np.zeros((9, 9)))

                remove_deadstones(b_pieces, w_pieces, move + 'B')
                b_pieces.append(sgf_to_array(move.sgf(board_size)))

            else:
                turns.append(np.ones([9, 9]))
                remove_deadstones(b_pieces, w_pieces, move + 'W')
                w_pieces.append(sgf_to_array(move.sgf(board_size)))
        if current_node.children:
            current_node = current_node.children[0]  # Move to the next node
        else:
            break
    return [b_pieces, w_pieces]
