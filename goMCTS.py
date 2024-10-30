import tensorflow as tf
import numpy as np
import copy
import math
from goEnv import GoGame  # Import GoGame from the appropriate module


class PreTrainedGoNetwork(tf.keras.Model):
    def __init__(self, checkpoint_path):
        super().__init__()
        self.num_nodes = 1024
        self.input_layer = tf.keras.layers.Input(shape=(83,))
        self.hidden_layer = tf.keras.layers.Dense(self.num_nodes, activation='relu')
        self.output_layer = tf.keras.layers.Dense(83, activation='softmax')

        # Load weights from the checkpoint
        self.load_weights(checkpoint_path)
    
    @tf.function
    def call(self, inputs):
        x = self.hidden_layer(inputs)
        return self.output_layer(x)
    
    def predict(self, board_state):
        network_input = self.prepare_input(board_state)
        return self(network_input)[0]

    def prepare_input(self, board_state):
        input_vector = np.zeros((1, 83))
        input_vector[0, :81] = board_state.flatten()
        return input_vector

class MCTSNode:
    def __init__(self, game_state, parent=None, move=None, prior=0.0):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = {}
        self.visits = 0
        self.value = 0
        self.prior = prior
        self.is_expanded = False

    def select_child(self, c_puct=1.0):
        best_score = float('-inf')
        best_child = None
        
        for move, child in self.children.items():
            if child.visits == 0:
                q_value = 0
            else:
                q_value = child.value / child.visits
                
            # UCB formula with prior probability
            u_value = c_puct * child.prior * math.sqrt(self.visits) / (1 + child.visits)
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_child = (move, child)
                
        return best_child

class MCTS:
    def __init__(self, network, num_simulations=800, c_puct=1.0):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def search(self, game_state):
        root = MCTSNode(game_state)
        for _ in range(self.num_simulations):
            node = root
            scratch_game = copy.deepcopy(game_state)
            
            while node.is_expanded and not scratch_game.is_game_over():
                move, node = node.select_child(self.c_puct)
                scratch_game.step(move)

            if not node.is_expanded and not scratch_game.is_game_over():
                policy = self.network.predict(scratch_game.board)
                valid_moves = scratch_game.get_legal_actions()
                for move in valid_moves:
                    move_idx = move[0] * 9 + move[1] if move != "pass" else 81
                    new_game = copy.deepcopy(scratch_game)
                    new_game.step(move)
                    node.children[move] = MCTSNode(new_game, node, move, policy[move_idx])
                node.is_expanded = True

            value = self.evaluate_terminal(scratch_game) if scratch_game.is_game_over() else self.evaluate_position(scratch_game)

            while node:
                node.visits += 1
                node.value += value
                node = node.parent

        visits = {move: child.visits for move, child in root.children.items()}
        return max(visits, key=visits.get)
    
    
    def evaluate_terminal(self, game_state):
        black_stones, white_stones = game_state.count_stones()
        return 1 if black_stones > white_stones else -1
    
    def evaluate_position(self, game_state):
        prediction = self.network.predict(game_state.board)
        # Use the resign probability as a value estimate
        return 2 * prediction[82] - 1  # Scale from [0,1] to [-1,1]

def self_play_training(network, num_games=100, mcts_simulations=800):
    mcts = MCTS(network, num_simulations=mcts_simulations)
    training_data = []
    
    for game_idx in range(num_games):
        print(f"Playing game {game_idx + 1}/{num_games}")
        game = GoGame(size=9)
        game_states = []
        mcts_policies = []
        
        while not game.is_game_over():
            current_state = copy.deepcopy(game.board)
            
            # Get MCTS policy
            root = MCTSNode(game)
            for _ in range(mcts_simulations):
                mcts.search(game)
            
            # Create policy vector from visit counts
            policy = np.zeros(83)  # 81 moves + PASS + RESIGN
            total_visits = sum(child.visits for child in root.children.values())
            for move, child in root.children.items():
                if move == "pass":
                    policy[81] = child.visits / total_visits
                else:
                    policy[move[0] * 9 + move[1]] = child.visits / total_visits
            
            # Store state and policy
            game_states.append(current_state)
            mcts_policies.append(policy)
            
            # Select and play move
            move = mcts.search(game)
            game.step(move)
        
        # Game is over - get outcome
        outcome = mcts.evaluate_terminal(game)
        
        # Add game data to training set
        for state, policy in zip(game_states, mcts_policies):
            training_data.append({
                'state': state,
                'policy': policy,
                'value': outcome
            })
        
        # Periodically update network
        if (game_idx + 1) % 10 == 0:
            train_network_on_data(network, training_data)
            training_data = []  # Clear buffer after training
    
    return network

def train_network_on_data(network, training_data):
    # Prepare batch data
    states = np.array([data['state'] for data in training_data])
    policies = np.array([data['policy'] for data in training_data])
    
    # Train in mini-batches
    batch_size = 32
    num_batches = len(training_data) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        
        batch_states = states[start_idx:end_idx]
        batch_policies = policies[start_idx:end_idx]
        
        loss = network.train_on_batch(batch_states, batch_policies)
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{num_batches}, Loss: {loss:.4f}")

if __name__ == "__main__":
    # Load pre-trained network
    checkpoint_path = "checkpoints/model.ckpt"
    network = PreTrainedGoNetwork()
    
    # Load weights
    network.load_weights(checkpoint_path)
    
    # Perform reinforcement learning through self-play
    improved_network = self_play_training(
        network,
        num_games=100,
        mcts_simulations=800
    )
    
    # Save the improved network
    improved_network.save("checkpoints/model_improved")  # Saves full model
