#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated for TensorFlow 2
Original Author: nubby
"""

import tensorflow as tf
import pickle
import numpy as np
import os
import glob

cwd = os.getcwd()
pickleRoot = os.path.join(cwd, 'pickles2')
mixedPickleRoot = os.path.join(cwd, 'pickles_mixed')
checkpointFile = os.path.join(cwd, 'checkpoints/model.keras')
csvFile = os.path.join(cwd, 'trainResults/trainResults.csv')

def loadPickle(pickleFile, dataset, labels):
    try:
        with open(pickleFile, 'rb') as f:
            print("Loading from ", pickleFile)
            saved = pickle.load(f)
            datasetNew = saved['dataset'].astype('float32')
            labelsNew = saved['labels'].astype('float32')
            del saved

            if len(dataset) == 0:
                dataset, labels = datasetNew, labelsNew
            else:
                dataset = np.concatenate((dataset, datasetNew))
                labels = np.concatenate((labels, labelsNew))

            print("Total so far - Dataset shape:", dataset.shape)
            print("Total so far - Labels shape:", labels.shape)
            return dataset, labels
    except Exception as e:
        print(f"Unable to load data from {pickleFile}: {e}")
        return dataset, labels

labels, dataset = [], []
for pickleFile in glob.glob(os.path.join(pickleRoot, "*.pickle")):
    dataset, labels = loadPickle(pickleFile, dataset, labels)
for mixedPickleFile in glob.glob(os.path.join(mixedPickleRoot, "*.pickle")):
    dataset, labels = loadPickle(mixedPickleFile, dataset, labels)

for i in range(5):  # Print the first 5 samples
    print(f"Sample {i} - Features: {dataset[i]}, Label: {labels[i]}")
    
    
def randomize(dataset, labels):
    permutation = np.random.permutation(len(dataset))
    return dataset[permutation], labels[permutation]

dataset, labels = randomize(dataset, labels)

data_size = len(dataset)
test_size = 10000
print(data_size)
print(test_size)
train_size = data_size - test_size
train_dataset, test_dataset = dataset[:train_size - 1], dataset[train_size:]
train_labels, test_labels = labels[:train_size - 1], labels[train_size:]


print('Training:', train_dataset.shape, train_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

# Hyperparameters
num_nodes = 1024
batch_size = 1000
num_steps = 200_000

# Model definition
class SimpleModel(tf.keras.Model):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(num_nodes, activation='relu')
        self.dense2 = tf.keras.layers.Dense(83)

    def call(self, x):
        x = self.dense1(x)
        return self.dense2(x)

model = SimpleModel()
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

@tf.function
def train_step(batch_data, batch_labels):
    with tf.GradientTape() as tape:
        logits = model(batch_data, training=True)
        loss = loss_fn(batch_labels, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

@tf.function
def compute_accuracy(predictions, labels):
    # Ensure shapes are consistent, then calculate accuracy
    print("----")
    print(predictions.shape)
    print(labels.shape)
    predictions = tf.argmax(predictions, axis=1)
    labels = tf.argmax(labels, axis=1)
    correct_preds = tf.equal(predictions, labels)
    return tf.reduce_mean(tf.cast(correct_preds, tf.float32)) * 100


csv = open(csvFile, 'w')
csv.write("step,loss,batch_accuracy,test_accuracy\n")

for step in range(num_steps):
    offset = (step * batch_size) % (train_size - batch_size)
    if offset < batch_size:
        train_dataset, train_labels = randomize(train_dataset, train_labels)
        print("Re-randomizing...")

    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]

    loss = train_step(batch_data, batch_labels)
    if step % 500 == 0:
        predictions = model(batch_data, training=False)
        train_accuracy = compute_accuracy(predictions, batch_labels)
        
        test_predictions = model(test_dataset, training=False)
        test_accuracy = compute_accuracy(test_predictions, test_labels)
        
        print(f"Step {step}: Loss = {loss.numpy():.4f}, Train Acc = {train_accuracy.numpy():.2f}%, Test Acc = {test_accuracy.numpy():.2f}%")
        csv.write(f"{step},{loss.numpy()},{train_accuracy.numpy()},{test_accuracy.numpy()}\n")

model.save(checkpointFile)
print(f"Model saved in file: {checkpointFile}")
csv.close()

# Analyzing resign/pass conditions
test_predictions = model(test_dataset, training=False)
resign_count = np.sum(test_predictions[:, 82] > 0.1)
pass_count = np.sum(test_predictions[:, 81] > 0.5)

print("Cases where RESIGN prediction is > 0.1 in test predictions:", resign_count)
print("Cases where PASS prediction is > 0.5 in test predictions:", pass_count)

resign_count_true = np.sum(test_labels[:, 82] == 1)
pass_count_true = np.sum(test_labels[:, 81] == 1)

print("Cases where RESIGN is true in test_labels:", resign_count_true)
print("Cases where PASS is true in test_labels:", pass_count_true)
