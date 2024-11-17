from pysgf import SGF
from typing import List, Tuple
import numpy as np
from goEnv import GoGame

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
    if not move:
        return "pass"
    return coor_map.get(move[0]), coor_map.get(move[1])

def move_to_ndarray(move: Tuple[int, int], board_size: int) -> np.ndarray:
    label = np.zeros(board_size**2 + 1)
    if move == "pass":
        label[-1] = 1
    else:
        index = move[0] * 9 + move[1]
        label[index] = 1
    return label

def get_states_old_format(file: str, features: List[np.ndarray], labels: List[np.ndarray]):
    current_node = SGF.parse_file(file)
    board_shape = current_node.board_size
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
                labels.append(next_parsed_move)
        if move:
            parsed_move = sgf_to_tuple(move.sgf(board_shape))
            game.step(parsed_move)
            feature = game.get_board().flatten()
          # feature = np.append(feature, 0) 
          # if parsed_move == "pass":
          #     feature[81] = 1
          # 
          # 
          # 
          # 
          # 
          #  
            if np.any(feature):
                features.append(feature)
            else:
                # Array is empty (the move was a pass)
                pass_array = np.zeros((1, 82))
                pass_array[81] = 1
                features.append(pass_array)
        if current_node.children:
            current_node = current_node.children[0]  # Move to the next node
        else:
            break
        
    
def get_states(file: str, features: List[np.ndarray], labels: List[np.ndarray]):
    """Retrieve all states in the game and return as a list of features"""
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



features, labels = [], []
get_states('data/Top50/nine/nine_strong/2015-11-27T12:56:05.370Z_tdgvxji07vuf.sgf', features, labels)
print(features[-1])
print(labels[-1])