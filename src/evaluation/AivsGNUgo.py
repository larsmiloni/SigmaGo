import copy
import os

import numpy as np
from src.ai.policy_network import PolicyNetwork
from src.mcts.goMCTS import MCTS, train_network_on_data, extract_features, map_turn_to_player
from src.game.goEnv import GoGame
import subprocess
from time import sleep
import sys
import matplotlib.pyplot as plt

coor_map = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7,
    'j': 8
}


def run_gnugo():
    # Start GNU Go subprocess
    gnugo_process = subprocess.Popen(
        ['gnugo', '--mode', 'gtp', '--chinese-rules', '--boardsize', '9', '--level', '10', '--komi', '5.5'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True  # This ensures the input/output are treated as text
    )
    return gnugo_process

def send_command(process, command):
    """
    Send a command to the GNU Go subprocess and return the response.

    Args:
        process (subprocess.Popen): The GNU Go subprocess.
        command (str): The command to send.
    """
    process.stdin.write(command + '\n')
    process.stdin.flush()
    
    # Read the first line
    response = process.stdout.readline().strip()
    print(f"response: {response}")


    print("=" in response)
    while "=" not in response:
        response = process.stdout.readline().strip()
        print(f"response: {response}")
    
    
    
    return response

def convert_move_to_goMCTS(gnugo_move: str):
    """
    Convert a GNU Go move to a GoMCTS move.

    Args:
        gnugo_move (str): The GNU Go move to convert.
    """
    print('----- converting move ------')
    print('gnu move ',gnugo_move)

    if gnugo_move == "= PASS":
        return "pass"
    elif len(gnugo_move) == 4 and gnugo_move[0] == '=':
        row_index= gnugo_move[3].lower()
        col_index = gnugo_move[2].lower()
        column = coor_map[col_index]
        row = abs(int(row_index)-9)
        print(f"converted to  {column}{row}")
        return (column, row)
    else:
        raise ValueError(f"Invalid GNU Go move format: {gnugo_move}")

def convert_move_to_gnugo(goMCTS_move):
    """
    Convert a GoMCTS move to a GNU Go move.

    Args:
        goMCTS_move (tuple): The GoMCTS move to convert
    """
    print('----- converting move to gnu go------')
    print('gomcts move ',goMCTS_move)
    if goMCTS_move == "pass":
        return "pass"
    else:
        print('index 0 value ', goMCTS_move[0])
        column = chr(goMCTS_move[0] + ord('A'))
        if column == 'I':
            column = 'J'
        row = abs(goMCTS_move[1] -9)
        print(f"converted to {column}{row}")
        return f"{column}{row}"

def apply_move(game, gnugo, move, is_gnugo_move=False, mcts_policies=None):
    if move != "pass":
        # Apply the move to the GoMCTS environment
        game.step(move)
        policy = np.zeros(83)
        move_idx = move[0] * 9 + move[1]
        policy[move_idx] = 1
        mcts_policies.append(policy)

        # Apply the move to the GNU Go environment
        if not is_gnugo_move:
            gnugo_move = convert_move_to_gnugo(move)
            print('white plays ',gnugo_move)
            send_command(gnugo, f'play white {gnugo_move}')
    else:
        # Apply the pass move to both environments
        game.step("pass")
        if not is_gnugo_move:
            send_command(gnugo, "play white pass")
    game.render_in_terminal()

def plot_winrate_over_games(gnugo_wins, goMCTS_wins, gnugo_wins_list, goMCTS_wins_list, filename="win_ratio.png"):
    """
    Plot the win rate of GNU Go and GoMCTS over games.

    Args:
        gnugo_wins (int): The number of wins by GNU Go.
        goMCTS_wins (int): The number of wins by GoMCTS.
        filename (str): The filename to save the plot to
    """
    total_games = gnugo_wins + goMCTS_wins
    gnugo_winrate = gnugo_wins / total_games
    goMCTS_winrate = goMCTS_wins / total_games

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [gnugo_winrate, gnugo_winrate], label="GNU Go")
    plt.plot([0, 1], [goMCTS_winrate, goMCTS_winrate], label="GoMCTS")
    plt.xlabel("Game")
    plt.ylabel("Win Ratio")
    plt.title("Win Ratio over Games")
    plt.legend()
    plt.savefig(filename)
    print(f"Plot saved to: {filename}")
    # Plotting AI vs Random wins over games
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, total_games + 1), goMCTS_wins_list, label="AI Wins", marker="o")
    plt.plot(range(1, total_games + 1), gnugo_wins_list, label="Gnu Go Wins", marker="x")
    plt.title("AI Wins vs GnuGo Wins Over Games")
    plt.xlabel("Games")
    plt.ylabel("Wins")
    plt.legend()
    plt.grid(True)
    plt.savefig("ai_vs_GNUgo.png") 
    return gnugo_winrate, goMCTS_winrate


    

def main(simulations=1):
    """
    Run the main evaluation loop for AI vs GNU Go.

    Args:
        simulations (int): The number of simulations to run.
    """
    # Initialize GNU Go and GoMCTS
    gnugo_wins = 0
    goMCTS_wins = 0
    training_data = []
    ai_wins_list = []
    gnoGo_wins_list = []

    for sim in range(simulations):
        print(f"\nSimulation {sim}/{simulations}")
        gnugo = run_gnugo()
        if sim> 0:
            network = PolicyNetwork(model_path=f"models/VN-R3-C64-2-GNU-GO-{sim-1}.pt")
        else:
            network = PolicyNetwork(model_path=f"./models/VN-R3-C64-2-GNU-GO-9-10-sims.pt")

        goMCTS = MCTS(network, num_simulations=50)
        game = GoGame(size=9)
        game.reset()  # Reset the game state
        game_states = []
        mcts_policies = []
        pass_count_gnu_go = 0
        pass_count_mcts = 0

        while not game.isGameOver:
            # Determine the current player
            turn = game.get_turn()  # 1 for Black, 2 for White
            current_player = map_turn_to_player(turn)

            # Store current state and current player
            current_state = copy.deepcopy(game.state)
            game_states.append((current_state, current_player))

            # Feature extraction
            features = extract_features(current_state, current_player)

            if current_player == 'black':
                # GNU Go (Black) makes a move
                sleep(1)  # Optional: Adjust as needed
                gnugo_move = send_command(gnugo, 'genmove black')
                print(f"GNU Go (Black) move: {gnugo_move}")

                # Convert and apply GNU Go move
                goMCTS_move = convert_move_to_goMCTS(gnugo_move)
                apply_move(game, gnugo, goMCTS_move, is_gnugo_move=True, mcts_policies=mcts_policies)

                if gnugo_move == " PASS":
                    print('GNO go passes pass count 1')
                    pass_count_gnu_go = 1
                    if pass_count_mcts and pass_count_gnu_go == 1:
                        #both has passed
                        game.isGameOver = True
                        break
                else:
                    pass_count_gnu_go = 0

            else:
                # GoMCTS (White) makes a move
                goMCTS_move = goMCTS.search(game)
                print(f"GoMCTS (White) move: {goMCTS_move}")

                # Convert and apply GoMCTS move
                apply_move(game, gnugo, goMCTS_move, is_gnugo_move=False, mcts_policies=mcts_policies)
                if goMCTS_move == "pass":
                    pass_count_mcts = 1
                    if pass_count_mcts and pass_count_gnu_go == 1:
                        #both has passed
                        game.isGameOver = True
                        break
                else:   
                    pass_count_mcts = 0

            # Check for game end conditions are handled within game.step()

        # Game over - evaluate result
        winner = game.determine_winner()
        print(f"\nGame finished!")
        print(f"Winner: {'black' if winner == 0 else 'white'}")
        if winner == 0:
            gnugo_wins += 1
        else:
            goMCTS_wins += 1

        ai_wins_list.append(goMCTS_wins)
        gnoGo_wins_list.append(gnugo_wins)

        # Collect training data
        for (state, player), policy in zip(game_states, mcts_policies):
            features = extract_features(state, player)
            training_data.append({
                'state': features,
                'policy': policy,
                'value': winner
            })

        # # Train the network
        train_network_on_data(network, training_data)
        network.save(path=f"./models/VN-R3-C64-2-GNU-GO-{sim}.pt")

    # Plot win rates
    gnugo_winrate, goMCTS_winrate = plot_winrate_over_games(gnugo_wins, goMCTS_wins, gnoGo_wins_list, ai_wins_list)
    print(f"GNU Go win ratio: {gnugo_winrate:.2f}")
    print(f"GoMCTS win ratio: {goMCTS_winrate:.2f}")

if __name__ == "__main__":
    """
    Run the main evaluation loop for AI vs GNU Go.

    SysArgs:
        simulations (int): The number of simulations to run.
    """
    os.makedirs('models', exist_ok=True)
    try:
        simulations = int(sys.argv[1])
    except (IndexError, ValueError):
        simulations = 1  # Default value if no argument is provided or if it's invalid
    main(simulations)