import tensorflow as tf
import numpy as np
import copy
import math
import os
from goEnv import GoGame  # Import GoGame from the appropriate module


class PreTrainedGoNetwork:
    def __init__(self, checkpoint_path):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Define the same network architecture as in trainCNN.py
            self.num_nodes = 1024
            
            # Placeholders
            self.input_layer = tf.placeholder(tf.float32, shape=(None, 83))
            
            # Variables with the same names as in original training
            self.weights_1 = tf.Variable(tf.truncated_normal([83, self.num_nodes]), name='weights_1')
            self.biases_1 = tf.Variable(tf.zeros([self.num_nodes]), name='biases_1')
            self.weights_2 = tf.Variable(tf.truncated_normal([self.num_nodes, 83]), name='weights_2')
            self.biases_2 = tf.Variable(tf.zeros([83]), name='biases_2')
            
            # Network architecture
            self.relu_layer = tf.nn.relu(tf.matmul(self.input_layer, self.weights_1) + self.biases_1)
            self.logits = tf.matmul(self.relu_layer, self.weights_2) + self.biases_2
            self.prediction = tf.nn.softmax(self.logits)
            
            # Training ops
            self.labels = tf.placeholder(tf.float32, shape=(None, 83))
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            )
            self.optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)
            
            # Initialize session and load checkpoint
            self.session = tf.Session(graph=self.graph)
            saver = tf.train.Saver()
            saver.restore(self.session, checkpoint_path)
            
    def predict(self, board_state):
        # Convert board state to network input format
        network_input = self.prepare_input(board_state)
        
        with self.graph.as_default():
            prediction = self.session.run(
                self.prediction,
                feed_dict={self.input_layer: network_input}
            )
        return prediction[0]
    
    def train_on_batch(self, states, labels):
        with self.graph.as_default():
            _, loss_value = self.session.run(
                [self.optimizer, self.loss],
                feed_dict={
                    self.input_layer: states,
                    self.labels: labels
                }
            )
        return loss_value
    
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
            
            # Selection
            while node.is_expanded and not scratch_game.is_game_over():
                move, node = node.select_child(self.c_puct)
                scratch_game.step(move)
            
            # Expansion
            if not node.is_expanded and not scratch_game.is_game_over():
                policy = self.network.predict(scratch_game.board)
                valid_moves = scratch_game.get_legal_actions()
                
                # Create children for all valid moves
                for move in valid_moves:
                    if move == "pass":
                        move_idx = 81  # PASS move index
                    else:
                        move_idx = move[0] * 9 + move[1]
                    
                    new_game = copy.deepcopy(scratch_game)
                    new_game.step(move)
                    node.children[move] = MCTSNode(
                        new_game,
                        parent=node,
                        move=move,
                        prior=policy[move_idx]
                    )
                node.is_expanded = True
            
            # Evaluation
            if scratch_game.is_game_over():
                value = self.evaluate_terminal(scratch_game)
            else:
                value = self.evaluate_position(scratch_game)
            
            # Backpropagation
            while node is not None:
                node.visits += 1
                node.value += value
                node = node.parent
        
        # Select move based on visit counts
        visits = {move: child.visits for move, child in root.children.items()}
        return max(visits.items(), key=lambda x: x[1])[0]
    
    
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
    network = PreTrainedGoNetwork(checkpoint_path)
    
    # Perform reinforcement learning through self-play
    improved_network = self_play_training(
        network,
        num_games=100,
        mcts_simulations=800
    )
    
    # Save improved network
    improved_network.save_model("checkpoints/model_improved.ckpt")