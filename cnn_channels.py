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

def move_to_ndarray(move: Tuple[int, int]) -> np.ndarray:
    label = np.zeros(9 * 9 + 1)
    if move == "pass":
        label[-1] = 1
    else:
        index = move[0] * 9 + move[1]
        label[index] = 1
    return label

def get_states(file: str, features: List[np.ndarray], labels: List[np.ndarray]):
    """Retrieve all states in the game and return as a list of features"""
    current_node = SGF.parse_file(file)

    board_size = (current_node.board_size)  # Use the board size from the root node
    game = GoGame(size=board_size[0])
    while current_node:
        
        move = current_node.move
        #Check if the current node has any children
        if current_node.children:
            next_node = current_node.children[0]
            next_move = next_node.move
             
            # Append the next move to labels
            if next_move:
                next_parsed_move = move_to_ndarray(sgf_to_tuple(next_move.sgf(board_size)))
                labels.append(next_parsed_move.T)
        if move:
            parsed_move = sgf_to_tuple(move.sgf(board_size))
            game.step(parsed_move)
            features.append(game.state) 
        if current_node.children:
            current_node = current_node.children[0]  # Move to the next node
        else:
            break


#features, labels = [], []
#get_states('data/Top50/nine/nine_strong/2015-11-27T12:56:05.370Z_tdgvxji07vuf.sgf', features, labels)
#
#print(len(features))
#print(len(labels))
#for label in labels:
#    if label[-1] == 1:
#            print("pass")
#    else:
#        label = np.delete(label, -1)
#        print(label.reshape((9, 9)).T)
    