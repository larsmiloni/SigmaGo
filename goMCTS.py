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



# Enable GPU memory growth to prevent TensorFlow from allocating all GPU memory at once
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s). GPU memory growth enabled.")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPUs found. Running on CPU.")

# Enable mixed precision training for better GPU performance
tf.keras.mixed_precision.set_global_policy('mixed_float16')

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
    def __init__(self, network, num_simulations=50, c_puct=math.sqrt(2), batch_size=32):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.batch_size = batch_size
        # Enable XLA compilation for faster execution
        if tf.test.is_built_with_cuda():
            self.network.compile(jit_compile=True)

    @tf.function(experimental_relax_shapes=True)
    def predict_batch(self, states):
        """GPU-optimized batch prediction"""
        return self.network(states, training=False)

    def search(self, game_state: GoGame) -> Union[Tuple[int, int], str]:
        if game_state.state[govars.DONE].any():
            print("Game is already over.")
            return "pass"
        
        try:
            legal_actions = game_state.get_legal_actions()
        except:
            print('Error getting legal actions')
            return "pass"
        
        if not legal_actions:
            return "pass"
        
        root = MCTSNode(game_state)
        
        # Batch states for GPU processing
        states_batch = []
        nodes_batch = []
        
        for sim in range(self.num_simulations):
            if sim % 50 == 0:
                print(f"Running simulation {sim}/{self.num_simulations}")
            
            node = root
            scratch_game = copy.deepcopy(game_state)
            
            # Selection phase
            while node.is_expanded and not scratch_game.state[govars.DONE].any():
                try:
                    move, node = node.select_child(self.c_puct)
                    if node is None:
                        return "pass"
                    
                    if move != "pass":
                        scratch_game.step(move)
                except KeyError:
                    move = "pass"
            
            # Expansion and evaluation phase
            if not node.is_expanded and not scratch_game.state[govars.DONE].any():
                states_batch.append(scratch_game.state)
                nodes_batch.append((node, scratch_game))
                
                # Process batch when it reaches batch_size
                if len(states_batch) >= self.batch_size:
                    self._process_batch(states_batch, nodes_batch)
                    states_batch = []
                    nodes_batch = []
            
            # Evaluation for terminal states
            value = (self.evaluate_terminal(scratch_game) if scratch_game.state[govars.DONE].any()
                    else self.evaluate_position(scratch_game))
            
            # Backup
            node.backup(value)
        
        # Process remaining states
        if states_batch:
            self._process_batch(states_batch, nodes_batch)
        
        # Select best move based on visit counts
        visits = {move: child.visits for move, child in root.children.items()}
        if not visits:
            return "pass"
        
        best_move = max(visits.items(), key=lambda x: x[1])[0]
        print(f"Visit counts for considered moves:")
        for move, visits in sorted(visits.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"- {chr(65 + move[1] if move[1] < 8 else 66 + move[1])}{9 - move[0]}: {visits} visits")
        
        return best_move

    def _process_batch(self, states_batch, nodes_batch):
        """Process a batch of states on GPU with expanded node handling"""
        if not states_batch:
            return
            
        # Convert to tensor for GPU processing
        states_tensor = tf.convert_to_tensor(states_batch)
        
        # Batch predict on GPU
        with tf.device('/GPU:0'):
            policy_preds = self.predict_batch(states_tensor)
        
        # Expand nodes with predictions
        for (node, scratch_game), policy_pred in zip(nodes_batch, policy_preds):
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
            
            # Evaluate position for the newly expanded node
            value = self.evaluate_position(scratch_game)
            node.backup(value)

    def evaluate_terminal(self, game_state: GoGame) -> int:
        """Evaluate terminal game state."""
        winner = game_state.determine_winner()
        return winner
    
    def evaluate_position(self, game_state: GoGame) -> float:
        """GPU-optimized position evaluation"""
        with tf.device('/GPU:0'):
            state_tensor = tf.convert_to_tensor([game_state.state])
            prediction = self.predict_batch(state_tensor)[0]
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
        mcts_simulations=20
    )

    # Save the improved network
    improved_network.save("models/PN_R3_C64_IMPROVED_MODEL.pt")  # Saves full model
    # except Exception as e:
    #     print("Unable to load pre-trained network:", e)
    #     print("Please ensure the checkpoint file exists and is valid.")
   
    
   
