import tensorflow as tf
from typing import List, Dict, Union, Tuple, Type
import numpy as np
import copy
import math
from MCTSnode import MCTSNode
import matplotlib.pyplot as plt
from goEnv import GoGame  # Import GoGame from the appropriate module
# from policy_network import PolicyNetwork
import govars



class PreTrainedGoNetwork(tf.keras.Model):
    def __init__(self, checkpoint_path):
        super().__init__()
        self.num_nodes = 1024
        self.input_layer = tf.keras.layers.Input(shape=(1, 83))
        self.hidden_layer = tf.keras.layers.Dense(self.num_nodes, activation='relu')
        self.output_layer = tf.keras.layers.Dense(83, activation='softmax')
        # Load weights from the checkpoint
        self.load_weights(checkpoint_path)
            
    def predict(self, board_state):
        network_input = self.prepare_input(board_state)
        return self(network_input)[0]
    
    @tf.function
    def call(self, inputs):
        x = self.hidden_layer(inputs)
        return self.output_layer(x)
    
    def prepare_input(self, board_state):
        # Convert 9x9 board to 83-length input vector
        # First 81 positions are the board state
        # Position 81 is for PASS
        # Position 82 is for RESIGN
        input_vector = np.zeros((1, 83))
        input_vector[0, :81] = board_state.flatten()
        # We'll set PASS and RESIGN to 0 by default
        return input_vector
    
    def save_model(self, checkpoint_path):
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.session, checkpoint_path)

    def select_move(self, game_state) -> Tuple[int, int]:
        """
        Selects a move based on the predicted policy.
        
        Args:
            game_state (GoGame): Current state of the game.
        
        Returns:
            tuple: Selected move as (row, col) or "pass"
        """
        policy = self.predict(game_state.state[govars.BOARD].flatten())
        legal_moves = game_state.get_legal_actions()
        
        # Initialize list for all legal moves including pass
        legal_move_probs = []
        
        # Handle regular moves
        for move in legal_moves:
            if move == "pass":
                # Assuming the last element in policy is for pass
                pass_prob = policy[-1]
                legal_move_probs.append((move, pass_prob))
            else:
                # Regular board position moves
                move_prob = policy[move[0] * 9 + move[1]]
                legal_move_probs.append((move, move_prob))
        
        # Sort by probability
        legal_move_probs.sort(key=lambda x: x[1], reverse=True)
        
        if not legal_move_probs:
            return "pass"
        
        return legal_move_probs[0][0]


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
        if game_state.isGameOver:
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
            while node.is_expanded and not scratch_game.isGameOver:
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
            if not node.is_expanded and not scratch_game.isGameOver:
                # Fixed: Properly unpack both policy and value predictions
                #Might need to change this to unpack the policy and value predictions
                policy_pred = self.network.predict(scratch_game.state[govars.BOARD].flatten())
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
        value = (self.evaluate_terminal(scratch_game) if scratch_game.isGameOver
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
        prediction = self.network.predict(game_state.state[govars.BOARD].flatten())
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
        
        while not game.isGameOver:
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
                # 'value': winner
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
    # values = np.array([data['value'] for data in training_data])
    
    # Train in mini-batches
    batch_size = 32
    num_batches = len(training_data) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        
        batch_states = states[start_idx:end_idx]
        batch_policies = policies[start_idx:end_idx]
        # batch_values = values[start_idx:end_idx]
        
        loss = network.train_on_batch(batch_states, batch_policies)
        # loss = network.train_on_batch(batch_states, batch_policies)
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{num_batches}, Loss: {loss:.4f}")

def print_board_state(game):
    game.render_in_terminal()


    


       
if __name__ == "__main__":
    # Load pre-trained network
    model_path = "models/model.keras"
    print("Initializing network...")
    try:
        network = PreTrainedGoNetwork(checkpoint_path=model_path)
        network.load_weights(model_path)
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
    assert network is not None, "Network not initialized."
    print("Starting self-play training...")
    improved_network = self_play_training(
        network=network,
        num_games=1,
        mcts_simulations=10
    )

    # Save the improved network
    improved_network.save("models/model_improved.keras")  # Saves full model
    
    
    
    # except Exception as e:
    #     print("Unable to load pre-trained network:", e)
    #     print("Please ensure the checkpoint file exists and is valid.")