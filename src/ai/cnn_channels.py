from pysgf import SGF
from typing import List, Tuple
import numpy as np
from src.game.goEnv import GoGame

coor_map = {
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

def sgf_to_tuple(move: str) -> tuple:
    """
    Convert an SGF move to a tuple of coordinates.

    Args:
        move (str): The SGF move to convert.

    Returns:
        tuple: The move as a tuple of coordinates.
    """
    if not move:
        return "pass"
    return coor_map.get(move[0]), coor_map.get(move[1])

def move_to_ndarray(move: Tuple[int, int], board_size: int) -> np.ndarray:
    """
    Convert a move to a one-hot encoded ndarray.

    Args:
        move (Tuple[int, int]): The move to convert.
        board_size (int): The size of the board.

    Returns:
        np.ndarray: The one-hot encoded move.
    """
    label = np.zeros(board_size**2 + 1)
    if move == "pass":
        label[-1] = 1
    else:
        index = move[0] * 9 + move[1]
        label[index] = 1
    return label        
    
def get_states(file: str, features: List[np.ndarray], labels: List[np.ndarray]):
    """
    Retrieve all states in the game and return as a list of features

    Args:
        file (str): The file to parse.
        features (List[np.ndarray]): The list to append features to.
        labels (List[np.ndarray]): The list to append labels to.
    """
    current_node = SGF.parse_file(file)

    board_shape = current_node.board_size  # Use the board size from the root node
    board_size = board_shape[0]
    game = GoGame(size=board_size)
    while current_node:
        
        move = current_node.move
        #Check if the current node has any children
        if current_node.children:
            next_node = current_node.children[0]
            next_move = next_node.move
             
            # Append the next move to labels
            if next_move:
                next_parsed_move = move_to_ndarray(sgf_to_tuple(next_move.sgf(board_shape)), board_size)
                labels.append(next_parsed_move.T)
        if move:
            parsed_move = sgf_to_tuple(move.sgf(board_shape))
            game.step(parsed_move)
            features.append(game.state) 
        if current_node.children:
            current_node = current_node.children[0]  # Move to the next node
        else:
            break

def get_states_new(file: str, features: List[np.ndarray], labels: List[np.ndarray]):
    current_node = SGF.parse_file(file)
    board_size = current_node.board_size
    game = GoGame(size=board_size[0])

    while current_node:
        move = current_node.move
        if move:
            parsed_move = sgf_to_tuple(move.sgf(board_size))
            current_label = move_to_ndarray(parsed_move, board_size[0])
            features.append(game.state.copy())
            labels.append(current_label)
            try:
                game.step(parsed_move)
            except Exception as e:
                print(f"Invalid move sequence detected: {e}")
        current_node = current_node.children[0] if current_node.children else None


"""
Uncoment to test
"""
# features, labels = [], []
# get_states('data/Top50/nine/nine_strong/2015-11-27T12:56:05.370Z_tdgvxji07vuf.sgf', features, labels)
# print(features[-1])
# print(labels[-1])