import tensorflow as tf
from typing import List, Dict, Union, Tuple, Type
import numpy as np
import copy
import math
from MCTSnode import MCTSNode
import matplotlib.pyplot as plt
from goEnv import GoGame  # Import GoGame from the appropriate module
from policy_network import PolicyNetwork
import govars
import torch
from torch import nn





def softmax(x):
    e_x = np.exp(x - np.max(x))  # Stability trick for softmax
    return e_x / e_x.sum()




class MCTS:
    """
    Implements Monte Carlo Tree Search for decision-making in Go. Uses a neural network to evaluate
    board positions and refine move selection through self-play.

    Attributes:
        network (tf.keras.Model): Pre-trained neural network model.
        num_simulations (int): Number of MCTS simulations per search.
        c_puct (float): Exploration-exploitation constant.
    """
    def __init__(self, network, num_simulations=50, c_puct=math.sqrt(2)):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def search(self, game_state: GoGame) -> Union[Tuple[int, int], str]:
        """
        Conducts a full search to select the best move for the current game state.
        Now includes board visualization after each move.

        Args:
            game_state (GoGame): The current game state to analyze.

        Returns:
            Union[Tuple[int, int], str]: The best move found by MCTS.
        """
        # First check if the game is already over
        if game_state.state[govars.DONE].any():
            print("Game is already over.")
            return "pass"
        
        # Get legal actions before starting search
        try:
            legal_actions = game_state.get_legal_actions()
        except:
            print('Game state:', game_state.state)
            print('Game state board:', game_state.board)
            
            print('Game state determine_winner:', game_state.determine_winner())
            print('Game state get_score:', game_state.get_score())
            print('Game state get_legal_actions:', game_state.get_legal_actions())

        
        print('Legal actions:', legal_actions)
        if not legal_actions:
            return "pass"
        
        root = MCTSNode(game_state)
    
        for sim in range(self.num_simulations):
            if sim % 50 == 0:
                print(f"Running simulation {sim}/{self.num_simulations}")
                
            node = root
            scratch_game = copy.deepcopy(game_state)

            while node.is_expanded and not scratch_game.state[govars.DONE].any():
                move, next_node = node.select_child(self.c_puct)

                    # Add a check in case `node` is None
                if next_node is None:
                    break # Reached a leaf node
                node = next_node
                scratch_game.step(move)
            
            if not node.is_expanded and not scratch_game.state[govars.DONE].any():
                # Get both policy and value predictions
                policy_pred, value_pred = self.network.predict(scratch_game.state)
                policy_pred = policy_pred.detach().cpu().numpy()
                value_pred = value_pred  # Already a scalar

                valid_moves = scratch_game.get_legal_actions()
                # Normalize policy predictions over valid moves
                policy_sum = np.sum([policy_pred[move_idx(move)] for move in valid_moves])
                for move in valid_moves:
                    idx = move_idx(move)
                    prior = policy_pred[idx] / policy_sum if policy_sum > 0 else 1 / len(valid_moves)
                    if move != "pass":
                        new_game = copy.deepcopy(scratch_game)
                        new_game.step(move)
                        node.children[move] = MCTSNode(
                            new_game,
                            parent=node,
                            move=move,
                            prior=prior
                        )
                    else:
                        node.children["pass"] = MCTSNode(
                            scratch_game,
                            parent=node,
                            move="pass",
                            prior=prior
                        )
                node.is_expanded = True

                # Backup the value
                node.backup(value_pred)
                continue  # Start next simulation
        
        # Select best move based on visit counts
        visits = {move: child.visits for move, child in root.children.items()}
        if not visits:
            return "pass"
        
        best_move = max(visits.items(), key=lambda x: x[1])[0]
        # print(f"\nSelected move: {chr(65 + best_move[1] if best_move[1] < 8 else 66 + best_move[1])}{9 - best_move[0]}")
        print(f"Visit counts for considered moves:")
        for move, visits in sorted(visits.items(), key=lambda x: x[1], reverse=True)[:5]:
            if move == "pass":
                print(f"- pass: {visits} visits")
            else:
                print(f"- {chr(65 + move[1] if move[1] < 8 else 66 + move[1])}{9 - move[0]}: {visits} visits")

        
        return best_move
    
    
    def evaluate_terminal(self, game_state: GoGame) -> int:
        """Evaluate terminal game state."""
        winner = game_state.determine_winner()
    
        return winner
    
    def evaluate_position(self, game_state: GoGame) -> float:
        """Evaluate non-terminal position using network prediction."""
        _, value_pred = self.network.predict(game_state.state)
        return value_pred  # Already scaled to [-1, 1]

def self_play_training(network: Type[tf.keras.Model], num_games=10, mcts_simulations=20, vsRandom=False):
    """
    Fixed version of self-play training that properly maintains game state
    """
    ai_win_count = 0
    random_win_count = 0
    ai_wins = []
    random_wins = []
    mcts = MCTS(network, num_simulations=mcts_simulations)
    training_data = []
    
    for game_idx in range(num_games):
        print(f"\n=== Playing game {game_idx + 1}/{num_games} ===")
        game = GoGame(size=9)
        game_states = []
        mcts_policies = []
        move_count = 0
        current_player = 'black'  # Track current player
        
        while not game.state[govars.DONE].any():
            print(f"\nMove {move_count + 1} - {current_player}'s turn")
            turn = game.get_turn()  # 1 for Black, 2 for White
            current_state = copy.deepcopy(game.state)
            current_player = map_turn_to_player(turn)
            # Store current state and current player
            game_states.append((current_state, current_player))
            features = extract_features(current_state, current_player)
            
            # Get move from MCTS
            if vsRandom:
                if current_player == 'black':
                    move = mcts.search(game)
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
            else:
                move = mcts.search(game)

            
            # Make the move and update game state
            if move != "pass":
                
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
                print(f"Move: {move}")
                
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
        print(f"\nGame {game_idx + 1} finished!")
        print(f"Winner: {winner}")
        
        # Add game data to training set
       
        for (state,player), policy in zip(game_states, mcts_policies):
            features = extract_features(state, player)
            training_data.append({
                'state': features,
                'policy': policy,
                'value': winner
            })
        
        
        print("\nUpdating network...")
        train_network_on_data(network, training_data)
        training_data = []
        
        if vsRandom:
            # AI plays as black, Random as white
            if winner == 0:  # Black wins
                ai_win_count += 1
                print("AI (Black) wins!")
            else:  # White wins
                random_win_count += 1
                print("Random (White) wins!")
    
        
        # Update win tracking lists
        ai_wins.append(ai_win_count)
        random_wins.append(random_win_count)
        #train the network
        if vsRandom:
            network.save(path=f"./models/VN-R3-C64-2-self-play{game_idx}.pt")
        else:
            network.save(path=f"./models/VN-R3-C64-2-play-random{game_idx}.pt")
    



    # Plotting AI vs Random wins over games
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_games + 1), ai_wins, label="AI Wins", marker="o")
    plt.plot(range(1, num_games + 1), random_wins, label="Random Wins", marker="x")
    plt.title("AI Wins vs Random Wins Over Games")
    plt.xlabel("Games")
    plt.ylabel("Wins")
    plt.legend()
    plt.grid(True)
    plt.savefig("ai_vs_random.png") 

    print(f"\nAI wins: {ai_win_count}")
    print(f"Random wins: {random_win_count}")
    
    

def train_network_on_data(network: Type[tf.keras.Model], training_data: List[Dict[str, np.ndarray]]):
    """
    Trains the network on generated data in mini-batches.
    
    Args
        network (tf.keras.Model): Model to train.
        training_data (List[Dict[str, np.ndarray]]): List of training samples with board states, policies, and outcomes.
    """
   
    states = np.array([data['state'] for data in training_data])
    policies = np.array([data['policy'] for data in training_data])
    values = np.array([data['value'] for data in training_data])

    # Convert to tensors
    states = torch.tensor(states, dtype=torch.float32).to(network.device)
    policies = torch.tensor(policies, dtype=torch.float32).to(network.device)
    values = torch.tensor(values, dtype=torch.float32).unsqueeze(1).to(network.device)

    print(f"States shape: {states.shape}")  # Should be [batch_size, 7, 9, 9]
    print(f"Policies shape: {policies.shape}")
    print(f"Values shape: {values.shape}")

    # Define loss functions
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

    # Train in mini-batches
    batch_size = 32
    num_samples = len(training_data)
    num_batches = num_samples // batch_size + int(num_samples % batch_size > 0)

    for epoch in range(1):  # Number of epochs
        print(f"\nEpoch {epoch + 1}/{1}")
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        for batch_idx in range(num_batches):
            batch_indices = indices[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            batch_states = states[batch_indices]
            batch_policies = policies[batch_indices]
            batch_values = values[batch_indices]

            optimizer.zero_grad()

            policy_out, value_out = network(batch_states)

            # Policy loss: cross-entropy between predicted policy and target policy
            policy_loss = policy_loss_fn(policy_out, torch.argmax(batch_policies, dim=1))

            # Value loss: mean squared error between predicted value and target value
            value_loss = value_loss_fn(value_out, batch_values)

            # Total loss
            loss = policy_loss + value_loss

            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")

def print_board_state(game):
    game.render_in_terminal()

def extract_features(state, current_player):
    """
    Convert the board state to input features for the network.

    Args:
        state (np.ndarray): The board state, shape [7,9,9], with values:
                            0 = empty,
                            1 = black stone,
                           -1 = white stone
        current_player (int): 1 for black, -1 for white

    Returns:
        np.ndarray: Input features, shape [7,9,9]
    """
    features = np.zeros((7, 9, 9), dtype=np.float32)

    # Feature 0: Current player's stones
    features[0] = (state[govars.BLACK] if current_player == 'black' else state[govars.WHITE]).astype(np.float32)

    # Feature 1: Opponent's stones
    features[1] = (state[govars.WHITE] if current_player == 'black' else state[govars.BLACK]).astype(np.float32)

    # Feature 2: Turn indicator (1 for current player, 0 otherwise)
    if current_player == 'white':
        features[2] = np.ones((9, 9), dtype=np.float32)
    else:
        features[2] = np.zeros((9, 9), dtype=np.float32)
    

    # Feature 3: Invalid moves
    features[3] = state[govars.INVD]

    # Feature 4: Previous move was a pass
    features[4] = state[govars.PASS]

    # Feature 5: Game over
    features[5] = state[govars.DONE]

    # Feature 6: Board state
    features[6] = state[govars.BOARD]

    return features

def map_turn_to_player(turn):
    """
    Maps the turn value from GoGame to current_player.

    Args:
        turn (int): The turn value from GoGame.get_turn().

    Returns:
        int: 1 for Black, -1 for White.
    """
    if turn == 1:
        return 'black'
    elif turn == 2:
        return 'white'
    else:
        raise ValueError(f"Invalid turn value: {turn}")


    
def move_idx(move, board_size=9):
    if move == "pass":
        return board_size * board_size  # Last index for "pass"
    else:
        return move[0] * board_size + move[1]


       
if __name__ == "__main__":
    torch.cuda.empty_cache()
    # Load pre-trained network
    model_path = "./models/VN-R3-C64-2.pt"
    print("Initializing network...")
    network = PolicyNetwork(model_path)

    
    
    # # Load weights
    # # try:
    # print("Loading pre-trained network...")
    # try:
    #     network.load_weights(model_path=model_path)
    #     print("Loaded pre-trained network.")
    # except:
    #     print("Unable to load pre-trained network.", e)

        # Perform reinforcement learning through self-play
    print("Starting self-play training...")
    self_play_training(
        network=network,
        num_games=5,
        mcts_simulations=5,
        vsRandom=True
    )

    
    
   
    
    # except Exception as e:
    #     print("Unable to load pre-trained network:", e)
    #     print("Please ensure the checkpoint file exists and is valid.")