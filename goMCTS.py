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
            # print('Is scratch game over:', scratch_game.isGameOver)
            # print('Is node expanded:', node.is_expanded)
            # Selection - use UCB to select moves until reaching a leaf
            while node.is_expanded and not scratch_game.state[govars.DONE].any():
                try:
                    move, node = node.select_child(self.c_puct)

                       # Add a check in case `node` is None
                    if node is None:
                        print("Warning: No valid child node selected, defaulting to 'pass'.")
                        return "pass"  # Exit the selection loop since no valid child is found


                    # Check if the move is valid in the children dictionary
                    if move in node.children:
                        node = node.children[move]
                    else:
                        move = "pass"
                except KeyError:
                    move = "pass"
                
                if move != "pass":
                    scratch_game.step(move)
            
            # Expansion - add child nodes for all legal moves
            if not node.is_expanded and not scratch_game.state[govars.DONE].any():
                # Fixed: Properly unpack both policy and value predictions
                #Might need to change this to unpack the policy and value predictions
                policy_pred = self.network.predict(scratch_game.state)
                valid_moves = scratch_game.get_legal_actions()
                
                for move in valid_moves:
                    if move != "pass":
                        move_idx = move[0] * 9 + move[1]
                        new_game = copy.deepcopy(scratch_game)
            
                        new_game.step(move)
                        node.children[move] = MCTSNode(
                            new_game, 
                            parent=node,
                            move=move,
                            prior=policy_pred[move_idx]
                        )
                node.is_expanded = True
        
        # Evaluation
        value = (self.evaluate_terminal(scratch_game) if scratch_game.state[govars.DONE].any()
                else self.evaluate_position(scratch_game))
        
        # Backup
        node.backup(value)
        
        # Select best move based on visit counts
        visits = {move: child.visits for move, child in root.children.items()}
        if not visits:
            return "pass"
        
        best_move = max(visits.items(), key=lambda x: x[1])[0]
        # print(f"\nSelected move: {chr(65 + best_move[1] if best_move[1] < 8 else 66 + best_move[1])}{9 - best_move[0]}")
        print(f"Visit counts for considered moves:")
        for move, visits in sorted(visits.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"- {chr(65 + move[1] if move[1] < 8 else 66 + move[1])}{9 - move[0]}: {visits} visits")
        
        return best_move
    
    
    def evaluate_terminal(self, game_state: GoGame) -> int:
        """Evaluate terminal game state."""
        winner = game_state.determine_winner()
    
        return winner
    
    def evaluate_position(self, game_state: GoGame) -> float:
        """Evaluate non-terminal position using network prediction."""
        prediction = self.network.predict(game_state.state)
        # Use the pass probability as a value estimate
        return 2 * prediction[81]  # Scale from [0,1] to [-1,1]

def self_play_training(network: Type[tf.keras.Model], num_games=10, mcts_simulations=200):
    """
    Fixed version of self-play training that properly maintains game state
    """
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
            current_state = copy.deepcopy(game.state)
            
            # Store current state before move
            game_states.append(current_state)
            
            # Get move from MCTS
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
       
        for state, policy in zip(game_states, mcts_policies):
            training_data.append({
                'state': state,
                'policy': policy,
                'value': winner
            })
        
        # Update network periodically
        if (game_idx + 1) % 5 == 0:
            print("\nUpdating network...")
            train_network_on_data(network, training_data)
            training_data = []
    
    return network

def train_network_on_data(network: Type[tf.keras.Model], training_data: List[Dict[str, np.ndarray]]):
    """
    Trains the network on generated data in mini-batches.
    
    Args:
        network (tf.keras.Model): Model to train.
        training_data (List[Dict[str, np.ndarray]]): List of training samples with board states, policies, and outcomes.
    """
   
    # Prepare batch data
    states = np.array([data['state'] for data in training_data])
    policies = np.array([data['policy'] for data in training_data])
    values = np.array([data['value'] for data in training_data])
    
    # Train in mini-batches
    batch_size = 32
    num_batches = len(training_data) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        
        batch_states = states[start_idx:end_idx]
        batch_policies = policies[start_idx:end_idx]
        batch_values = values[start_idx:end_idx]
        
        loss = network.train_on_batch(batch_states, [batch_policies, batch_values])
        # loss = network.train_on_batch(batch_states, batch_policies)
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{num_batches}, Loss: {loss:.4f}")

def print_board_state(game):
    game.render_in_terminal()


    


       
if __name__ == "__main__":
    # Load pre-trained network
    model_path = "models/PN-R3-C64.pt"
    print("Initializing network...")
    try:
        network = PolicyNetwork(model_path=model_path)
    except Exception as e:
        print("Unable to load pre-trained network:", e)
        print("Please ensure the checkpoint file exists and is valid.")
    
    
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
    improved_network = self_play_training(
        network=network,
        num_games=1,
        mcts_simulations=10
    )

    # Save the improved network
    improved_network.save("models/PN_R3_C64_IMPROVED_MODEL.pt")  # Saves full model
    
    network = PolicyNetwork(model_path="models/PN_R3_C64_IMPROVED_MODEL.pt")
    
    improved_network_delux = self_play_training(
        network=network,
        num_games=1,
        mcts_simulations=5
    )

    # Save the improved network
    improved_network_delux.save("models/PN_R3_C64_IMPROVED_MODEL_2.pt")  # Saves full model
    
    # except Exception as e:
    #     print("Unable to load pre-trained network:", e)
    #     print("Please ensure the checkpoint file exists and is valid.")