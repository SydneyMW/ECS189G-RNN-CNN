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

# test_set and train_set: element_spec = (
#   TensorSpec(shape=(None,), 
#   dtype=tf.string,
#   name=None),
#   TensorSpec(shape=(None,),
#   dtype=tf.int32,
#   name=None))

BUFFER_SIZE = 10000
BATCH_SIZE = 100
## SHUFFLE DATASETS
train_dataset = train_set.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_set.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

## IDENTIFY MOST FREQUENT VOCABULARY
VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(train_set.map(lambda text, label: text))

# vocab = np.array(encoder.get_vocabulary())
# vocab[:20] :  vocab[:20]
# array(['', '[UNK]', 'the', 'and', 'a', 'of', 'to', 'is', 'in', 'it', 'i',
#      'this', 'that', 'br', 'was', 'as', 'for', 'with', 'movie', 'but'],
#     dtype='<U14')

## ENCODE MOST FREQUENT VOCABULARY
encoded_example = encoder(example)[:3].numpy()
# encoded_example
# array([[  2,   1,   1, ...,   0,   0,   0],
#        [  1,   1,   7, ...,   0,   0,   0],
#        [  4,  50, 315, ...,   0,   0,   0]], dtype=int64)

## CREATE RNN MODEL 
# bidirectional for classification
model = tf.keras.Sequential([
    
    # first layer is encoder
    encoder, 

    # next layer is embedding layer
    tf.keras.layers.Embedding( 
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    
    # next layers use LSTM to avoid gradient explosion, relu activation
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    
    # output layer has one neuron with classification (0/1)
    tf.keras.layers.Dense(1)
])

# Train model with Adam optimization, accuracy loss metric
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(train_set, epochs=10,
                    validation_data=test_set,
                    validation_steps=30)

test_loss, test_acc = model.evaluate(test_dataset)

## DETERMINE ACCURACY AND LOSS
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

## PLOT EPOCH CONVERGENCE OVER TIME
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_graphs(history, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
plt.ylim(0, None)