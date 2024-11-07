from typing import Tuple
import numpy as np
import glob
import os
import pickle
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt

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

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        self.to(self.device)

    def forward(self, x):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.batch_norm1(out)
        out = self.relu1(out)
        #out = self.pad2(out)
        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = out + x
        out = self.relu2(out)
        return out

class PolicyNetwork(torch.nn.Module):

    def __init__(self, alpha, model_path, num_res=3, num_channel=3):
        super(PolicyNetwork, self).__init__()

        #self.input_channels = num_channel
        self.model_path = model_path
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
        # define optimizer
        self.optimizer = torch.optim.Adam(lr=alpha, params=self.parameters())
        self.loss = torch.nn.BCELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        self.to(self.device)

    def load_weights(self):
        self.load_weights(self.model_path)
    
    def predict(self, board_state) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts the move probabilities for a given board state.
        
        Args:
            board_state (np.ndarray): Current state of the Go board, as a 9x9x6 array.

        Returns:
            tf.Tensor: Probability distribution over possible moves.
        """
        network_input = self.prepare_input(board_state)
        return self(network_input)[0]

    def prepare_input(self, board_state) -> np.ndarray:
        """
        Converts board state to network input format.

        Args:
            board_state (np.ndarray): Current state of the board, as a 9x9x7 array.

        Returns:
            np.ndarray: Prepared input vector for the network.
        """
        # Ensure the input is in the correct shape (batch_size, height, width, channels)
        input_vector = np.expand_dims(board_state, axis=0)
    
        return input_vector

    def define_network(self):
        #policy head
        self.policy_conv = torch.nn.Conv2d(self.num_channel, 2, kernel_size=1)
        self.policy_batch_norm = torch.nn.BatchNorm2d(2)
        self.relu = torch.nn.ReLU()
        self.policy_fc1 = torch.nn.Linear(2*9*9, 128)
        self.policy_fc2 = torch.nn.Linear(128, 82)
        self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()

        # network start
        #self.pad = torch.nn.Pad(1)
        self.conv = torch.nn.Conv2d(self.state_channel, self.num_channel, kernel_size=1)
        self.batch_norm = torch.nn.BatchNorm2d(self.num_channel)
        self.relu = torch.nn.ReLU()

        for i in range(1, self.num_res+1):
            self.res_block["r"+str(i)] = Block(self.num_channel)

    def forward(self, x):
        out = torch.Tensor(x).float().to(self.device)

        #out = self.pad(out)
        out = self.conv(out)
        out = self.batch_norm(out)
        out = self.relu(out)

        for i in range(1, self.num_res+1):
            out = self.res_block["r"+str(i)](out)

        # policy head
        out = self.policy_conv(out)
        out = self.policy_batch_norm(out)
        out = self.relu(out)
        out = out.reshape(-1, 2*9*9)
        out = self.policy_fc1(out)
        out = self.relu(out)
        out = self.policy_fc2(out)
        out = self.softmax(out)
        return out


    def optimize(self, x, y, x_t, y_t, batch_size=16, iterations=10, alpha=0.1, test_interval=1000, save=False):
        x_t = x_t.float().to(self.device)
        y_t = y_t.float().to(self.device)

        # Model and log paths
        model_name = "PN-R" + str(self.num_res) + "-C" + str(self.num_channel)
        self.model_name = model_name
        model_path = "models/policy_net/{}".format(model_name)
        log_path = "logs/policy_net/{}/".format(model_name)

        num_batch = x.shape[0] // batch_size

        print("Training Model:", model_name)

        # Training loop
        for iter in tqdm(range(iterations)):
            training_loss_sum = 0.0
            correct_predictions = 0
            total_predictions = 0

            for i in range(num_batch):
                batch_x = x[i*batch_size:(i+1)*batch_size].float().to(self.device)
                batch_y = y[i*batch_size:(i+1)*batch_size].float().to(self.device)

                prediction = self.forward(batch_x)
                loss = self.loss(prediction, batch_y)
                training_loss_sum += loss.item()

                # Calculate training accuracy for the batch
                correct_predictions += (prediction.argmax(dim=1) == batch_y.argmax(dim=1)).sum().item()
                total_predictions += batch_y.size(0)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                torch.cuda.empty_cache()

                self.historical_loss.append(loss.detach())

                # Test model at specified intervals
                if (iter * num_batch + i) % test_interval == 0:
                    self.test_model(x_t, y_t)
                    self.test_iteration.append(iter * num_batch + i)

            # Calculate average training loss and training accuracy for the iteration
            avg_training_loss = training_loss_sum / num_batch
            training_accuracy = correct_predictions / total_predictions

            # Print training loss, training accuracy, and latest test accuracy after each iteration
            print(f"Iteration {iter + 1}/{iterations}")
            print(f"Training Loss: {avg_training_loss:.4f}")
            print(f"Training Accuracy: {training_accuracy * 100:.2f}%")
            if self.test_accuracies:
                print(f"Test Accuracy: {self.test_accuracies[-1] * 100:.2f}%")
            torch.cuda.empty_cache()
        
        self.save()
            

    def test_model(self, x_t, y_t):

        prediction = self.forward(x_t)
        test_accuracy = self.get_test_accuracy(prediction, y_t)
        test_loss = self.get_test_loss(prediction, y_t)

        m = len(self.historical_loss)
        l = torch.Tensor(self.historical_loss)
        training_loss = torch.sum(l)/m

        del(prediction)
        torch.cuda.empty_cache()
        self.historical_loss = []

        self.test_accuracies.append(test_accuracy.detach().type(torch.float16))
        self.test_losses.append(test_loss.detach().type(torch.float16))
        self.training_losses.append(training_loss.type(torch.float16))


    def get_test_accuracy(self, prediction, y_t):

        c = torch.zeros(y_t.shape[0], y_t.shape[1], device=prediction.device)

        c[prediction == prediction.max(dim=0)[0]] = 1
        c[prediction != prediction.max(dim=0)[0]] = 0

        correct_percent = torch.sum(c*y_t) / y_t.shape[0]

        return correct_percent

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


def load_features_labels(test_size: int):

    cwd = os.getcwd()
    pickleRoot = os.path.join(cwd, 'pickles')
    mixedPickleRoot = os.path.join(cwd, 'pickles_mixed')

    features, labels = [], []

    def loadPickle(pickleFile, features, labels):
        try:
            with open(pickleFile, 'rb') as f:
                print("Loading from ", pickleFile)
                saved = pickle.load(f)
                datasetNew = saved['dataset'].astype('float32')
                labelsNew = saved['labels'].astype('float32')
                del saved

                if len(features) == 0:
                    features, labels = datasetNew, labelsNew
                else:
                    features.append(datasetNew)
                    labels.append(labelsNew)

                print("Total so far - Features shape:", features.shape)
                print("Total so far - Labels shape:", labels.shape)
                return features, labels
        except Exception as e:
            print(f"Unable to load data from {pickleFile}: {e}")
        return features, labels

    for pickleFile in glob.glob(os.path.join(pickleRoot, "*.pickle")):
        features, labels = loadPickle(pickleFile, features, labels)
    for mixedPickleFile in glob.glob(os.path.join(mixedPickleRoot, "*.pickle")):
        features, labels = loadPickle(mixedPickleFile, features, labels)

    data_size = len(features)
    train_size = data_size - test_size
    x_train, x_test = features[:train_size], features[train_size:]
    y_train, y_test = labels[:train_size], labels[train_size:]

    return torch.Tensor(x_train), torch.Tensor(y_train), torch.Tensor(x_test), torch.Tensor(y_test)

def main():
    pn = PolicyNetwork(alpha=0.01, num_res=3, num_channel=64)
    x_train, y_train, x_test, y_test = load_features_labels(1000)
    pn.optimize(x_train, y_train, x_test, y_test, batch_size=1_000, iterations=1_000, save=True)
    print(pn.forward(x_train).shape)
    plt.plot(pn.historical_loss)
    plt.show()


if __name__ == "__main__": main()