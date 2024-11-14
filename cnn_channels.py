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
    