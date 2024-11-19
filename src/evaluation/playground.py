from matplotlib import pyplot as plt
from src.ai.policy_network import PolicyNetwork
import ast
from src.game.goEnv import GoGame
from typing import Type
from src.mcts.goMCTS import MCTS
import copy
import numpy as np
import src.utils.govars as govars

def ai_vs_ai():
    """
    Simulates a game between two AI players.
    """
    print('Slect AI mode:')
    print('1. Ai👾 with MCTS vs Ai👾 with MCTS')
    print('2. Ai👾 with MCTS vs Ai👾 with CNN policy')
    print('3. Ai👾 with CNN policy vs Ai👾 with CNN policy')
    print('4. Ai👾 with CNN policy vs Random🎲')
    print('5. Ai👾 with MCTS vs Random🎲')

    mode = input()
    if mode == '1':

        ai_vs_ai_mcts()
    elif mode == '2':
        ai_vs_ai_mcts_cnn()
    elif mode == '3':
        ai_vs_ai_cnn()
    elif mode == '4':
        ai_cnn_vs_random()
    elif mode == '5':
        ai_mcts_vs_random()

def ai_vs_ai_mcts():
    """
    Simulates a game between two AI players using MCTS.
    """
    network_player1 = PolicyNetwork("./models/PN_R3_C64_IMPROVED_MODEL.pt")
    network_player2 = PolicyNetwork("./models/PN_R3_C64_IMPROVED_MODEL_2.pt") 
    create_ai_match(network_player1, network_player2, isBothMCTS=True)

def ai_vs_ai_mcts_cnn():
    """
    Simulates a game between two AI players using MCTS and CNN.
    """
    network_player1 = PolicyNetwork("./models/PN_R3_C64.pt")
    network_player2 = PolicyNetwork("./models/PN_R3_C64_IMPROVED_MODEL.pt") 
    create_ai_match(network_player1, network_player2, isCNNvsMCTS=True)

def human_vs_ai():
    """
    Simulates a game between a human player and an AI player
    """
    network = PolicyNetwork("./models/PN_R3_C64.pt")
    create_ai_match(network, None, isHumanvsAi=True)

def ai_vs_human():
    """
    Simulates a game between an AI player and a human player
    """
    network = PolicyNetwork("./models/PN_R3_C64.pt")
    create_ai_match(network, None, isAivsHuman=True)

def ai_vs_ai_cnn():
    """
    Simulates a game between two AI players using CNN.
    """
    network_player1 = PolicyNetwork("./models/PN_R3_C64.pt")
    network_player2 = PolicyNetwork("./models/VN-R3-C64-150-iter.pt") 
    create_ai_match(network_player1, network_player2, isBothCNN=True)

def ai_mcts_vs_random():
    """
    Simulates a game between an AI player using MCTS and a random player.
    """
    network = PolicyNetwork("./models/10-sims/VN-R3-C64-2-GNO-GO-9-10-sims.pt")
    create_ai_match(network, None, isMCTSvsRandom=True, games=5)

def ai_cnn_vs_random():
    """
    Simulates a game between an AI player using CNN and a random player.
    """
    network = PolicyNetwork("./models/50-sims/VN-R3-C64-2-GNO-GO-4-50-sims.pt")
    create_ai_match(network, None, isCNNvsRandom=True, games=50)

def create_ai_match(network_player1: Type[PolicyNetwork], network_player2: Type[PolicyNetwork],
                    isHumanvsAi=False, isBothCNN=False, isBothMCTS=False,
                    isCNNvsMCTS=False, isMCTSvsCNN=False, isAivsHuman=False,
                    isCNNvsRandom=False, isMCTSvsRandom=False, games=1):
    """
    Simulates a game between two AI players or an AI and a human player.
    
    Args:
        network_player1 (PolicyNetwork): Pre-trained neural network model for player 1.
        network_player2 (PolicyNetwork): Pre-trained neural network model for player 2.
        isHumanvsAi (bool): True if the game is between a human and an AI player.
        isBothCNN (bool): True if both players are using CNN.
        isBothMCTS (bool): True if both players are using MCTS.
        isCNNvsMCTS (bool): True if player 1 is using CNN and player 2 is using MCTS.
        isMCTSvsCNN (bool): True if player 1 is using MCTS and player 2 is using CNN.
        isAivsHuman (bool): True if the game is between an AI and a human player.
        isCNNvsRandom (bool): True if player 1 is using CNN and player 2 is random.
        isMCTSvsRandom (bool): True if player 1 is using MCTS and player 2 is random.
        games (int): Number of games to simulate.
    """
    
    ai_win_count = 0
    random_win_count = 0
    ai_wins = []
    random_wins = []

    def play_game():
        """
        Simulates a single game between two AI players or an AI and a human player.
        """
        nonlocal ai_win_count, random_win_count

        mcts_simulations = 40
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
            mcts_player1 = MCTS(network_player1, num_simulations=mcts_simulations)
        elif isCNNvsRandom:
            cnn_player1 = PolicyNetwork(model_path=network_player1)
        elif isMCTSvsRandom:
            mcts_player1 = MCTS(network_player1, num_simulations=mcts_simulations)
        else:
            # Default
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
                    move = ast.literal_eval(input('Enter move (example: 2,4): ').replace('.', ','))
                        
            elif isHumanvsAi:
                print('Human vs AI')
                if current_player == 'black':
                    print('legal moves:', game.get_legal_actions())
                    move = ast.literal_eval(input('Enter move (example 2,4): '))
                else:
                    move = mcts_player1.search(game)
            elif isCNNvsRandom:
                print('AI vs Random')
                if current_player == 'black':
                    print('AI is thinking...')
                    move = cnn_player1.select_move(game)
                else:
                    print('Random turn')
                    # Select random legal move
                    legal_moves = game.get_legal_actions()
                    
                    if not legal_moves:
                        move = "pass"
                    else:
                        # Select a random move from the list of legal moves
                        selected_move = np.random.choice(len(legal_moves))
                        move = legal_moves[selected_move]
            elif isMCTSvsRandom:
                print('AI vs Random')
                if current_player == 'black':
                    print('AI is thinking...')
                    move = mcts_player1.search(game)
                else:
                    print('Random turn')
                    # Select random legal move
                    legal_moves = game.get_legal_actions()
                    
                    if not legal_moves:
                        move = "pass"
                    else:
                        # Select a random move from the list of legal moves
                        selected_move = np.random.choice(len(legal_moves))
                        move = legal_moves[selected_move]
            
            # Make the move and update game state
            if move != "pass":
                # Make the move
                game.step(move)  
                # Create policy vector from the move
                policy = np.zeros(83)
                move_idx = move[0] * 9 + move[1]
                policy[move_idx] = 1
                mcts_policies.append(policy)
                
                # Print updated board
                print("\nBoard after move:")
                print_board_state(game)
                
                # Switch players
                current_player = 'white' if current_player == 'black' else 'black'
                move_count += 1
                
                print('Move count:', move_count)
            else:
                print(f"{current_player} passes")
                game.step("pass")
                current_player = 'white' if current_player == 'black' else 'black'
        
        # Game over - evaluate result
        winner = game.determine_winner()
        print(f"\nGame finished!")
        print(f"Winner: {'black' if winner == 0 else 'white'}")
        if isCNNvsRandom or isMCTSvsRandom:
            # AI plays as black, Random as white
            if winner == 0:  # Black wins
                ai_win_count += 1
                print("AI (Black) wins!")
            else:  # White wins
                random_win_count += 1
                print("Random (White) wins!")
    
    for i in range(games):
        print(f"\nGame {i + 1}")
        play_game()
        
        # Update win tracking lists
        ai_wins.append(ai_win_count)
        random_wins.append(random_win_count)
        #train the network
    



    # Plotting AI vs Random wins over games
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, games + 1), ai_wins, label="AI Wins", marker="o")
    plt.plot(range(1, games + 1), random_wins, label="Random Wins", marker="x")
    plt.title("AI Wins vs Random Wins Over Games")
    plt.xlabel("Games")
    plt.ylabel("Wins")
    plt.legend()
    plt.grid(True)
    plt.savefig("ai_vs_random.png") 

    print(f"\nAI wins: {ai_win_count}")
    print(f"Random wins: {random_win_count}")
        

def print_board_state(game: type[GoGame]):
    """
    Print the current board state of the game.
    
    Args:
        game (GoGame): The game object.
    """
    game.render_in_terminal()

if __name__ == '__main__':
    """
    Select the mode to play the game.
    """

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






