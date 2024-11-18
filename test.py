import numpy as np
import torch
from policy_network import PolicyNetwork

def test_prepare_input():
    # Create a mock board state with shape (9, 9, 7)
    board_state = np.random.rand(7, 9, 9)

    # Initialize the PolicyNetwork
    model = PolicyNetwork()

    # Call the prepare_input method
    input_tensor = model.prepare_input(board_state)

    # Check the shape of the output tensor
    expected_shape = (1, 7, 9, 9)
    assert input_tensor.shape == expected_shape, f"Expected shape {expected_shape}, but got {input_tensor.shape}"

    print(f"Test passed! Output shape: {input_tensor.shape}")

if __name__ == "__main__":
    test_prepare_input()