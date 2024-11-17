from typing import Tuple
import numpy as np
import glob
import os
import pickle
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch import nn
from datetime import datetime
from time import sleep

class Block(torch.nn.Module):

    def __init__(self, num_channel):
        super(Block, self).__init__()
        self.pad1 = torch.nn.ZeroPad2d(1)
        self.conv1 = torch.nn.Conv2d(num_channel, num_channel, kernel_size=2)
        self.batch_norm1 = torch.nn.BatchNorm2d(num_channel)
        self.relu1 = torch.nn.ReLU()
        self.pad2 = torch.nn.ZeroPad2d(1)
        self.conv2 = torch.nn.Conv2d(num_channel, num_channel, kernel_size=2)
        self.batch_norm2 = torch.nn.BatchNorm2d(num_channel)
        self.relu2 = torch.nn.ReLU()
        print("Block Created")

    def forward(self, x):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.batch_norm1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = out + x
        out = self.relu2(out)
        return out

class PolicyNetwork(torch.nn.Module):

    def __init__(self, model_path=None, alpha=0.01, num_res=3, num_channel=3):
        super(PolicyNetwork, self).__init__()
        torch.cuda.empty_cache()
        print("Policy Network Created 1")
        #self.input_channels = num_channel
        self.model_path = model_path
        print('Model path:', self.model_path)
        self.state_channel = 7
        self.num_res = num_res
        self.res_block = torch.nn.ModuleDict()
        self.num_channel = num_channel
        self.historical_loss = []

        # network metrics
        self.training_losses = []
        self.test_losses = []
        self.training_accuracies = []
        self.test_accuracies = []
        self.test_iteration = []

        self.model_name = "VN-R" + str(self.num_res) + "-C" + str(self.num_channel)

        self.define_network()

        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        print("Using device pn:", self.device)
        self.to(self.device)

        try:
            if self.model_path:
                print("Loading model from:", self.model_path)
                self.load_state_dict(torch.load(self.model_path, map_location=self.device))
        except Exception as e:
            print(e)


        # define optimizer
        self.optimizer = torch.optim.Adam(lr=alpha, params=self.parameters())
        self.policy_loss_fn = torch.nn.CrossEntropyLoss()
        self.value_loss_fn = torch.nn.MSELoss()


    # def load_weights(self, model_path):

        # if self.device != 'cpu:0':

        # self.load_weights(model_path)

        # else:
        #     self.load_state_dict(torch.load(self.model_path, map_location=self.device))
    
        

    def predict(self, board_state):
        """
        Predicts the move probabilities for a given board state.

        Args:
            board_state (np.ndarray): Current state of the Go board, as a 9x9x6 array.

        Returns:
            tf.Tensor: Probability distribution over possible moves.
        """
        
        network_input = self.prepare_input(board_state)
        policy_out, value_out = self(network_input)
        probabilities = torch.softmax(policy_out[0], dim=0)
        value = value_out[0].item()
        return probabilities, value

    def prepare_input(self, board_state):
        """
        Converts board state to network input format.

        Args:
            board_state (np.ndarray): Current state of the board, as a 9x9x7 array.

        Returns:
            np.ndarray: Prepared input vector for the network.
        """
        # Ensure the input is in the correct shape (batch_size, height, width, channels)
        # Ensure the input is in the correct shape (batch_size, height, width, channels)
        input_vector = np.expand_dims(board_state, axis=0)  # [1, 9, 9, 7]
        input_tensor = torch.tensor(input_vector, dtype=torch.float32).to(self.device)
        return input_tensor

    def define_network(self):
        print("Defining network start...")
        # Network start
        self.conv = nn.Conv2d(self.state_channel, self.num_channel, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(self.num_channel)
        self.network_relu = nn.ReLU()
        print("Network start defined.")

        print("Defining residual blocks...")
        for i in range(1, self.num_res + 1):
            self.res_block["r" + str(i)] = Block(self.num_channel)
        print("Residual blocks defined.")

        print("Defining policy head...")
        # Policy head
        self.policy_conv = nn.Conv2d(self.num_channel, 2, kernel_size=1)
        self.policy_batch_norm = nn.BatchNorm2d(2)
        self.policy_relu = nn.ReLU()
        self.policy_fc1 = nn.Linear(2 * 9 * 9, 128)
        self.policy_fc2 = nn.Linear(128, 82)
        print("Policy head defined.")

        print("Defining value head...")
        # Value head
        self.value_conv = nn.Conv2d(self.num_channel, 1, kernel_size=1)
        self.value_batch_norm = nn.BatchNorm2d(1)
        self.value_relu = nn.ReLU()
        self.value_fc1 = nn.Linear(1 * 9 * 9, 128)
        self.value_fc2 = nn.Linear(128, 1)
        self.tanh = nn.Tanh()
        print("Value head defined.")

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected input to be a torch.Tensor but got {type(x)}")
        
        if x.device != self.device:
            x = x.to(self.device)
        out = x
        out = self.conv(out)
        out = self.batch_norm(out)
        out = self.network_relu(out)
        for i in range(1, self.num_res + 1):
            out = self.res_block["r" + str(i)](out)
        
        # Save the output from residual blocks
        res_out = out

        # Policy head
        policy_out = self.policy_conv(res_out)
        policy_out = self.policy_batch_norm(policy_out)
        policy_out = self.policy_relu(policy_out)
        policy_out = policy_out.reshape(-1, 2 * 9 * 9)
        policy_out = self.policy_fc1(policy_out)
        policy_out = self.policy_relu(policy_out)
        policy_out = self.policy_fc2(policy_out)

        # Value head
        value_out = self.value_conv(res_out)
        value_out = self.value_batch_norm(value_out)
        value_out = self.value_relu(value_out)
        value_out = value_out.reshape(-1, 1 * 9 * 9)
        value_out = self.value_fc1(value_out)
        value_out = self.value_relu(value_out)
        value_out = self.value_fc2(value_out)
        value_out = self.tanh(value_out)  # Output value in range [-1, 1]

        return policy_out, value_out

    def optimize(self, x, y_policy, y_value, x_t, y_t_policy, y_t_value, batch_size=64, iterations=10, alpha=0.001, test_interval=10, save=False):
        print("Optimizing")
        # Update optimizer with new learning rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)

        num_batch = x.shape[0] // batch_size

        for iter in tqdm(range(iterations)):
            training_loss_sum = 0.0
            correct_predictions = 0
            total_predictions = 0

            for i in range(num_batch):
                batch_x = x[i*batch_size:(i+1)*batch_size].float().to(self.device)
                batch_y_policy = y_policy[i*batch_size:(i+1)*batch_size].long().to(self.device)
                batch_y_value = y_value[i*batch_size:(i+1)*batch_size].float().to(self.device)

                policy_out, value_out = self.forward(batch_x)

                # Policy loss
                policy_loss = self.policy_loss_fn(policy_out, batch_y_policy)

                # Value loss
                value_loss = self.value_loss_fn(value_out.squeeze(), batch_y_value)

                # Total loss
                loss = policy_loss + value_loss

                training_loss_sum += loss.item()

                # Calculate training accuracy for the batch
                _, predicted = torch.max(policy_out.data, 1)
                correct_predictions += (predicted == batch_y_policy).sum().item()
                total_predictions += batch_y_policy.size(0)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            avg_training_loss = training_loss_sum / num_batch
            training_accuracy = correct_predictions / total_predictions

            print(f"Iteration {iter + 1}/{iterations}")
            print(f"Training Loss: {avg_training_loss:.4f}")
            print(f"Training Accuracy: {training_accuracy * 100:.2f}%")

            # Test model at the end of each iteration
            if (iter + 1) % test_interval == 0:
                self.test_model(x_t, y_t_policy, y_t_value)
                self.test_iteration.append(iter)

                if self.test_accuracies:
                    print(f"Test Accuracy: {self.test_accuracies[-1] * 100:.2f}%")
                torch.cuda.empty_cache()

        if save:
            self.save()



    def test_model(self, x_t, y_t_policy, y_t_value, batch_size=32):
        total_correct = 0
        total_loss = 0
        num_batches = x_t.shape[0] // batch_size + (1 if x_t.shape[0] % batch_size != 0 else 0)

        with torch.no_grad():
            for i in range(num_batches):
                batch_x = x_t[i*batch_size:(i+1)*batch_size].float().to(self.device)
                batch_y_policy = y_t_policy[i*batch_size:(i+1)*batch_size].long().to(self.device)
                batch_y_value = y_t_value[i*batch_size:(i+1)*batch_size].float().to(self.device)

                policy_out, value_out = self.forward(batch_x)

                # Policy loss
                policy_loss = self.policy_loss_fn(policy_out, batch_y_policy)

                # Value loss
                value_loss = self.value_loss_fn(value_out.squeeze(), batch_y_value)

                # Total loss
                loss = policy_loss + value_loss

                total_loss += loss.item() * batch_x.size(0)

                # Accuracy
                _, predicted = torch.max(policy_out.data, 1)
                correct = (predicted == batch_y_policy).sum().item()
                total_correct += correct

        test_accuracy = total_correct / x_t.shape[0]
        test_loss = total_loss / x_t.shape[0]
        self.test_accuracies.append(test_accuracy)
        self.test_losses.append(test_loss)
        print(datetime.now().strftime("%H:%M:%S"))


    def get_test_accuracy(self, prediction, y_t):
        _, predicted = torch.max(prediction.data, 1)
        correct = (predicted == y_t).sum().item()
        return correct


    def get_test_loss(self, prediction, y_t):
        return self.loss(prediction, y_t)

    def save_training_loss(self, path=""):
        file_name = path + self.model_name+"-Train-Loss.csv"
        file = open(file_name, "w")
        file.write("iteration,loss\n")
        for iteration, loss in enumerate(self.training_losses):
            file.write("{},{}\n".format(iteration, loss))
        file.close()

    def save_test_loss(self, path=""):
        assert len(self.test_losses) == len(self.test_iteration)
        file_name = path + self.model_name+"-Test-Loss.csv"
        file = open(file_name, "w")
        file.write("iteration,loss\n")
        for i, loss in enumerate(self.test_losses):
            file.write("{},{}\n".format(self.test_iteration[i], loss))
        file.close()

    def save_test_accuracy(self, path=""):
        assert len(self.test_accuracies) == len(self.test_iteration)
        file_name = path + self.model_name+"-Test-Accuracy.csv"
        file = open(file_name, "w")
        file.write("iteration,accuracy\n")
        for i, acc in enumerate(self.test_accuracies):
            file.write("{},{}\n".format(self.test_iteration[i], acc))
        file.close()

    def save_metrics(self):
        self.save_training_loss()
        self.save_test_loss()
        self.save_test_accuracy()

    def save(self):
        torch.save(self.state_dict(), self.model_name+".pt")

    def get_edge_mask(self, board_size):
        mask = torch.zeros(board_size, board_size)
        mask[0, :] = 1     # Top edge
        mask[-1, :] = 1    # Bottom edge
        mask[:, 0] = 1     # Left edge
        mask[:, -1] = 1    # Right edge
        mask = mask.flatten()
        # Append zero for the 'pass' move
        mask = torch.cat([mask, torch.tensor([0.0])])
        return mask

    def select_move(self, game_state) -> Tuple[int, int]:
        policy = self.predict(game_state.state).detach().cpu().numpy()
        legal_moves = game_state.get_legal_actions()

        # Apply edge penalty
        edge_mask = self.get_edge_mask(board_size=9).numpy()
        penalty_weight = 0.5  # Adjust this value as needed

        # Reduce probabilities for edge moves
        policy -= penalty_weight * edge_mask

        # Ensure probabilities are non-negative
        policy = np.maximum(policy, 0)

        # Re-normalize the probabilities
        total_prob = np.sum(policy)
        if total_prob > 0:
            policy /= total_prob
        else:
            # If all probabilities are zero, assign uniform probability to legal moves
            policy = np.zeros_like(policy)
            for move in legal_moves:
                if move == "pass":
                    policy[-1] = 1.0
                else:
                    policy[move[0] * 9 + move[1]] = 1.0
            policy /= np.sum(policy)

        # Proceed with selecting the move
        legal_move_probs = []
        for move in legal_moves:
            if move == "pass":
                pass_prob = policy[-1]
                legal_move_probs.append((move, pass_prob))
            else:
                move_prob = policy[move[0] * 9 + move[1]]
                legal_move_probs.append((move, move_prob))

        # Sort by probability
        legal_move_probs.sort(key=lambda x: x[1], reverse=True)

        if not legal_move_probs:
            return "pass"

        return legal_move_probs[0][0]

def load_features_labels(test_size_ratio: float):
    if not (0.0 < test_size_ratio < 1.0):
        raise ValueError("test_size_ratio must be a float between 0 and 1.")

    cwd = os.getcwd()
    pickleRoot = os.path.join(cwd, 'pickles')
    mixedPickleRoot = os.path.join(cwd, 'pickles_mixed')

    all_features, all_labels = [], []

    def load_pickle(pickle_file):
        try:
            with open(pickle_file, 'rb') as f:
                print("Loading from", pickle_file)
                saved = pickle.load(f)
                dataset_new = saved['dataset'].astype('float32')
                labels_new = saved['labels'].astype('float32')
                return dataset_new, labels_new
        except Exception as e:
            print(f"Unable to load data from {pickle_file}: {e}")
            return None, None

    # Load pickles from both directories
    i = 0
    for pickle_file in glob.glob(os.path.join(pickleRoot, "*.pickle")):
        i += 1
        if i == 40:
            break
        features, labels = load_pickle(pickle_file)
        if features is not None and labels is not None:
            all_features.append(features)
            all_labels.append(labels)

    for mixed_pickle_file in glob.glob(os.path.join(mixedPickleRoot, "*.pickle")):
        features, labels = load_pickle(mixed_pickle_file)
        if features is not None and labels is not None:
            all_features.append(features)
            all_labels.append(labels)

    # Combine all features and labels into single arrays
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Generate value targets (set to zero for now)
    value_targets = np.zeros(len(features), dtype=np.float32)

    # Calculate the test size and split into training and testing sets
    data_size = len(features)
    test_size = int(data_size * test_size_ratio)
    train_size = data_size - test_size

    x_train, x_test = features[:train_size], features[train_size:]
    y_train_policy, y_test_policy = labels[:train_size], labels[train_size:]
    y_train_value, y_test_value = value_targets[:train_size], value_targets[train_size:]

    return (
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train_policy, dtype=torch.float32),
        torch.tensor(y_train_value, dtype=torch.float32),
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test_policy, dtype=torch.float32),
        torch.tensor(y_test_value, dtype=torch.float32),
    )


def main():
    sleep(5)
    pn = PolicyNetwork(alpha=0.01, num_res=3, num_channel=64)
    x_train, y_train_policy, y_train_value, x_test, y_test_policy, y_test_value = load_features_labels(0.2)

    # Convert policy labels to LongTensor for CrossEntropyLoss
    y_train_policy = torch.argmax(y_train_policy, dim=1).long()
    y_test_policy = torch.argmax(y_test_policy, dim=1).long()

    # Ensure value targets are FloatTensor
    y_train_value = y_train_value.float()
    y_test_value = y_test_value.float()

    pn.optimize(
        x_train,
        y_train_policy,
        y_train_value,
        x_test,
        y_test_policy,
        y_test_value,
        batch_size=128,
        iterations=150,
        save=True
    )
    plt.plot(pn.historical_loss)
    plt.show()
    pn.save_metrics()



if __name__ == "__main__": main()