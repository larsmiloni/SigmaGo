import numpy as np
import tensorflow as tf
import unittest
from goMCTS import MCTS  # Replace with actual import
from trainCNN import CNNModel  # Updated import

class GoDataGenerator:
    """Generate synthetic Go game data for testing."""
    @staticmethod
    def generate_random_board_state():
        """Generates a random board state as a 9x9 grid flattened into 83-element input."""
        board = np.random.choice([0, 1, -1], size=(9, 9))  # Random -1, 0, 1 for empty, black, white stones
        input_vector = np.zeros((1, 83))
        input_vector[0, :81] = board.flatten()  # Fill the 81 board positions
        return input_vector

    @staticmethod
    def generate_random_policy():
        """Generates a random policy vector for a board state."""
        policy = np.random.rand(83)
        return policy / policy.sum()  # Normalize to create a probability distribution

class TestCNNandMCTS(unittest.TestCase):
    def setUp(self):
        """Set up the pretrained network and MCTS with synthetic data."""
        self.network = CNNModel(checkpoint_path='checkpoints/model.ckpt')
        self.mcts = MCTS(self.network, num_simulations=100)

    def test_cnn_forward_pass(self):
        """Tests CNN forward pass with synthetic data."""
        board_state = GoDataGenerator.generate_random_board_state()
        predictions = self.network.predict(board_state)
        self.assertEqual(predictions.shape, (83,))
        self.assertTrue(np.all(predictions >= 0) and np.all(predictions <= 1),
                        "Predictions should be valid probabilities.")

    def test_mcts_search(self):
        """Tests MCTS search on synthetic board state."""
        game_state = GoDataGenerator.generate_random_board_state()
        best_move = self.mcts.search(game_state)
        self.assertIsNotNone(best_move, "MCTS should return a best move.")

    def test_training_data_format(self):
        """Validates the format of training data generated."""
        board_state = GoDataGenerator.generate_random_board_state()
        policy = GoDataGenerator.generate_random_policy()
        self.assertEqual(board_state.shape, (1, 83))
        self.assertEqual(policy.shape, (83,))
        self.assertAlmostEqual(np.sum(policy), 1.0, places=5, 
                               msg="Policy vector should be a valid probability distribution.")

if __name__ == "__main__":
    unittest.main()
