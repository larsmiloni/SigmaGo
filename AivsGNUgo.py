from policy_network import PolicyNetwork
from OnelayerMCTS import PreTrainedGoNetwork
from OnelayerMCTS import MCTS
from goEnv import GoGame
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
        ['gnugo', '--mode', 'gtp', '--chinese-rules', '--boardsize', '9', '--level', '0', '--komi', '7.5'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True  # This ensures the input/output are treated as text
    )
    return gnugo_process

def send_command(process, command):
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

def convert_move_to_goMCTS(gnugo_move):
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

def apply_move(game, gnugo, move, is_gnugo_move=False):
    if move != "pass":
        # Apply the move to the GoMCTS environment
        game.step(move)

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

def plot_winrate_over_games(gnugo_wins, goMCTS_wins, filename="win_ratio.png"):
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
    return gnugo_winrate, goMCTS_winrate


    

def main(simulations=1):
    # Initialize GNU Go and GoMCTS
    gnugo_wins = 0
    goMCTS_wins = 0
    for _ in range(simulations):
        gnugo = run_gnugo()
        network = PreTrainedGoNetwork(checkpoint_path="./models/model.keras")
        network.load_weights("./models/model.keras") 

        goMCTS = MCTS(network, num_simulations=5)
        goPolicy = network
        game = GoGame(size=9)

        while game.isGameOver==False:
            # Get move from GNU Go
        
            sleep(1)
            gnugo_move = send_command(gnugo, 'genmove black')
            print(f"GNU Go move: {gnugo_move}")

            

            # Apply GNU Go move to the GoMCTS environment
            goMCTS_move = convert_move_to_goMCTS(gnugo_move)
            apply_move(game, gnugo, goMCTS_move, is_gnugo_move=True)

            # Get move from GoMCTS
            goMCTS_move = goPolicy.select_move(game)
            print(f"GoMCTS move: {goMCTS_move}")

            
            

            # Apply GoMCTS move to the GNU Go environment
            apply_move(game, gnugo, goMCTS_move, is_gnugo_move=False)

            # Check for game end conditions
            
            if gnugo_move == 'resign' or goMCTS_move == 'resign':
                print("Game over")
                break

        # Game over - evaluate result
        winner = game.determine_winner()
        print(f"\nGame finished!")
        print(f"Winner: {'black' if winner == 0 else 'white'}")
        if winner == 0:
            gnugo_wins += 1
        else:
            goMCTS_wins += 1
            
    gnugo_winrate, goMCTS_winrate = plot_winrate_over_games(gnugo_wins, goMCTS_wins)
    print(f"GNU Go win ratio: {gnugo_winrate:.2f}")
    print(f"GoMCTS win ratio: {goMCTS_winrate:.2f}")

if __name__ == "__main__":
    try:
        simulations = int(sys.argv[1])
    except (IndexError, ValueError):
        simulations = 1  # Default value if no argument is provided or if it's invalid
    main(simulations)