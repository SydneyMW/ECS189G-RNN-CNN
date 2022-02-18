import sys
sys.path.append('C:/Users/Sydney/Desktop/ECS 189G/Project')

import numpy as np
import collections
import pathlib
import matplotlib.pyplot as plt # for plotting epochs
import os # for walking through directory

import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import utils
#from tensorflow.python.keras.layers import TextVectorization

import tensorflow_datasets as tfds
import tensorflow_text as tf_text
tfds.disable_progress_bar()

def plot_graphs(history, metric): # graph plotting function
    plt.plot(history.history[metric])
    plt.plot(history.hitory['val'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])

c_train_dir = 'data/stage_4_data/text_classification/train'
c_test_dir = 'data/stage_4_data/text_classification/test'
c_train_words = []
c_test_words = []
batch_size = 100

train_set = tf.keras.utils.text_dataset_from_directory(
    c_train_dir,
    batch_size = batch_size
) # Found 25000 training files belonging to 2 classes

test_set = tf.keras.utils.text_dataset_from_directory(
    c_test_dir,
    batch_size=batch_size
) # Found 25000 testing files belonging to 2 classes.

VOCAB_SIZE = 10000
binary_vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='binary')

MAX_SEQUENCE_LENGTH = 250
int_vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH)

train_text = train_set.map(lambda text, labels: text)
binary_vectorize_layer.adapt(train_text)
int_vectorize_layer.adapt(train_text)

def binary_vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return binary_vectorize_layer(text), label

def int_vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return int_vectorize_layer(text), label

binary_train_ds = train_set.map(binary_vectorize_text)
binary_test_ds = test_set.map(binary_vectorize_text)

int_train_ds = train_set.map(int_vectorize_text)
int_test_ds = test_set.map(int_vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE

def configure_dataset(dataset):
  return dataset.cache().prefetch(buffer_size=AUTOTUNE)

binary_train_ds = configure_dataset(binary_train_ds)
binary_test_ds = configure_dataset(binary_test_ds)

int_train_ds = configure_dataset(int_train_ds)
int_test_ds = configure_dataset(int_test_ds)

# Bag of Words Linear Model
binary_model = tf.keras.Sequential([layers.Dense(4)])
binary_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])
history = binary_model.fit(
    binary_train_ds, epochs=10)
