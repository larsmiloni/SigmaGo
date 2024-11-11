from goMCTS import create_ai_match
from policy_network import PolicyNetwork
import ast
from goEnv import GoGame
from typing import Type
import tensorflow as tf
from goMCTS import MCTS
import copy
import numpy as np
import govars

def ai_vs_ai():
    print('Slect AI mode:')
    print('1. Ai👾 with MCTS vs Ai👾 with MCTS')
    print('2. Ai👾 with MCTS vs Ai👾 with CNN policy')
    print('3. Ai👾 with CNN policy vs Ai👾 with CNN policy')
    mode = input()
    if mode == '1':

        ai_vs_ai_mcts()
    elif mode == '2':
        ai_vs_ai_mcts_cnn()
    elif mode == '3':
        ai_vs_ai_cnn()

def ai_vs_ai_mcts():
    network_player1 = PolicyNetwork("./models/PN_R3_C64_IMPROVED_MODEL.pt")
    network_player2 = PolicyNetwork("./models/PN_R3_C64_IMPROVED_MODEL_2.pt") 
    create_ai_match(network_player1, network_player2, isBothMCTS=True)

def ai_vs_ai_mcts_cnn():
    network_player1 = PolicyNetwork("./models/PN_R3_C64.pt")
    network_player2 = PolicyNetwork("./models/PN_R3_C64_IMPROVED_MODEL.pt") 
    create_ai_match(network_player1, network_player2, isCNNvsMCTS=True)

def human_vs_ai():
    network = PolicyNetwork("./models/PN_R3_C64.pt")
    create_ai_match(network, None, isHumanvsAi=True)

def ai_vs_human():
    network = PolicyNetwork("./models/PN_R3_C64.pt")
    create_ai_match(network, None, isAivsHuman=True)

def ai_vs_ai_cnn():
    network_player1 = PolicyNetwork("./models/PN_R3_C64.pt")
    network_player2 = PolicyNetwork("./models/PN_R3_C64_IMPROVED_MODEL.pt") 
    create_ai_match(network_player1, network_player2, isBothCNN=True)

def create_ai_match(network_player1: Type[tf.keras.Model], network_player2:Type[tf.keras.Model],
                     isHumanvsAi=False, isBothCNN=False, isBothMCTS=False,
                       isCNNvsMCTS=False, isMCTSvsCNN=False, isAivsHuman=False):
    """
    Simulates a game between two Ai players.
    
    Args:
        network_player1 (tf.keras.Model): Pre-trained neural network model for player 1.
        network_player12 (tf.keras.Model): Pre-trained neural network model for player 2.
    
    Returns:
        Type[tf.keras.Model]: AI player model.
    """


    mcts_simulations = 20
    if isBothMCTS:
        mcts_player1 = MCTS(network_player1, num_simulations=mcts_simulations)
        mcts_player2 = MCTS(network_player2, num_simulations=mcts_simulations)
    elif isBothCNN:
        cnn_player1 = PolicyNetwork(model_path=network_player1)
        cnn_player2 = PolicyNetwork(model_path=network_player2)
    elif isCNNvsMCTS:
        cnn_player1 = PolicyNetwork(model_path=network_player1)
        mcts_player2 = MCTS(network_player2, num_simulations=mcts_simulations)
    elif isMCTSvsCNN:
        mcts_player1 = MCTS(network_player1, num_simulations=mcts_simulations)
        cnn_player2 = PolicyNetwork(model_path=network_player2)
    elif isAivsHuman or isHumanvsAi:
        print('Preapering AI')
        mcts_player1 = MCTS(network_player1, num_simulations=mcts_simulations)
    else:
        #Default
        mcts_player1 = MCTS(network_player1, num_simulations=mcts_simulations)
        mcts_player2 = MCTS(network_player2, num_simulations=mcts_simulations)

    game = GoGame(size=9)
    game_states = []
    mcts_policies = []
    move_count = 0
    current_player = 'black'  # Track current player
    
    while not game.state[govars.DONE].any():
        print(f"\nMove {move_count + 1} - {current_player}'s turn")
        current_state = copy.deepcopy(game.state)
        
        # Store current state before move
        game_states.append(current_state)
        
        if isBothMCTS:
            if current_player == 'black':
                move = mcts_player1.search(game)
            else:
                move = mcts_player2.search(game)
        elif isBothCNN:
            if current_player == 'black':
                move = cnn_player1.select_move(game)
            else:
                move = cnn_player2.select_move(game)
        elif isCNNvsMCTS:
            if current_player == 'black':
                move = cnn_player1.select_move(game)
            else:
                move = mcts_player2.search(game)
        elif isMCTSvsCNN:
            if current_player == 'black':
                move = mcts_player1.search(game)
            else:
                move = cnn_player2.select_move(game)
        elif isAivsHuman:
            print('AI vs Human')
            print('Current player:', current_player)
            if current_player == 'black':
                print('AI is thinking...')
                move = mcts_player1.search(game)
            else:
                print('Human turn')
                print('legal moves:', game.get_legal_actions())
                move = ast.literal_eval(input('Enter move (example: 2,4): '))
                    
        elif isHumanvsAi:
            print('Human vs AI')
            if current_player == 'black':
                print('legal moves:', game.get_legal_actions())
                move = ast.literal_eval(input('Enter move (example 2,4): '))
            else:
                move = mcts_player1.search(game)
                    


        
        # Make the move and update game state
        if move != "pass":
            print('Move:', move)
            #Make the move
            game.step(move)  
            # Create policy vector from the move
            policy = np.zeros(83)
            move_idx = move[0] * 9 + move[1]
            policy[move_idx] = 1
            mcts_policies.append(policy)
            
            # Print updated board
            print("\nBoard after move:")
            print_board_state(game)
            print(f"Move: {chr(65 + move[1] if move[1] < 8 else 66 + move[1])}{9 - move[0]}")
            
            # Switch players
            current_player = 'white' if current_player == 'black' else 'black'
            move_count += 1
            
            print('Move count:', move_count)

        
        else:
            print(f"{current_player} passes")
            move = game.step("pass")
            current_player = 'white' if current_player == 'black' else 'black'
    
    # Game over - evaluate result
    winner = game.determine_winner()
    print(f"\nGame finished!")
    print(f"Winner: {'white' if winner == 1 else 'black'}") # Current player is the winner

def print_board_state(game):
    game.render_in_terminal()

if __name__ == '__main__':
    print('Slect mode:')
    print('1. Play human👨 vs AI👾')
    print('2. Play AI👾 vs AI👾')
    mode = input()
    if mode == '1':
        print('Select player:')
        print('1. Human👨 first')
        print('2. AI👾 first')
        player = input()
        if player == '1':
            human_vs_ai()
        else:
            ai_vs_human()
        
    else:
        ai_vs_ai()






