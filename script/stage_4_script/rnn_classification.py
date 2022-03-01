import sys
sys.path.append('C:/Users/Sydney/Desktop/ECS 189G/Project')

import numpy as np
import matplotlib.pyplot as plt # for plotting epochs
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import utils
import tensorflow_datasets as tfds
import tensorflow_text as tf_text
tfds.disable_progress_bar()

c_train_dir = 'data/stage_4_data/text_classification/train'
c_test_dir = 'data/stage_4_data/text_classification/test'

# # Directory for testing on smaller datasets (10 training, 10 testing, 5 neg/5 pos each)
# c_train_dir = 'data/stage_4_data/classification_small/train'
# c_test_dir = 'data/stage_4_data/classification_small/test'

c_train_words = []
c_test_words = []
batch_size = 200
seed = 42

raw_classification_train_ds = tf.keras.utils.text_dataset_from_directory(
    c_train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed) # Found 25000 training files belonging to 2 classes

raw_classification_val_ds = tf.keras.utils.text_dataset_from_directory(
    c_train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)

raw_classification_test_ds = tf.keras.utils.text_dataset_from_directory(
    c_test_dir,
    batch_size=batch_size) # Found 25000 testing files belonging to 2 classes.

VOCAB_SIZE = 10000
binary_vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='binary')
binary_vectorize_layer.adapt(raw_classification_train_ds.map(lambda text, label: text))

vocab = np.array(binary_vectorize_layer.get_vocabulary())
classification_train_mapped = raw_classification_train_ds.map(tf.autograph.experimental.do_not_convert(lambda text, labels: text))
binary_vectorize_layer.adapt(classification_train_mapped)

def classification_encoded_text(text, label):
  text = tf.expand_dims(text, -1)
  return binary_vectorize_layer(text), label

classification_train_ds = raw_classification_train_ds.map(classification_encoded_text)
classification_val_ds = raw_classification_val_ds.map(classification_encoded_text)
classification_test_ds = raw_classification_test_ds.map(classification_encoded_text)

AUTOTUNE = tf.data.AUTOTUNE

def configure_dataset(dataset):
  return dataset.cache().prefetch(buffer_size=AUTOTUNE)

classification_train_ds = configure_dataset(classification_train_ds)
classification_val_ds = configure_dataset(classification_val_ds)
classification_test_ds = configure_dataset(classification_test_ds)


# BAG OF WORDS METHOD 1 FOR BINARY MODEL
binary_model = tf.keras.Sequential([layers.Dense(4)])
binary_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])
history = binary_model.fit(
    classification_train_ds, epochs=300)
# 98.98% accuracy for binary with sparse categorical cross entropy, adam optimize, accuracy metric, 200 epoch


# EVALUATE TRAINING PROCESS
binary_loss, binary_accuracy = binary_model.evaluate(classification_test_ds)
print("Linear model on binary vectorized data:")
print(binary_model.summary())
print("Binary model accuracy: {:2.2%}".format(binary_accuracy))  # Binary model accuracy: 83.76%



# TEST OUT MANUALLY BY PROCESSING RAW STRINGS
export_model = tf.keras.Sequential(
    [binary_vectorize_layer, binary_model,
     layers.Activation('sigmoid')])

export_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer='adam',
    metrics=['accuracy'])

# Test it with `raw_classification_test_ds`, which yields raw strings
loss, accuracy = export_model.evaluate(raw_classification_test_ds)
print("Accuracy: {:2.2%}".format(binary_accuracy)) # 83%

# Run on raw strings
def get_string_labels(predicted_scores_batch):
  predicted_int_labels = tf.argmax(predicted_scores_batch, axis=1)
  predicted_labels = tf.gather(raw_classification_train_ds.class_names, predicted_int_labels)
  return predicted_labels

inputs = [
    "this movie is terribe one of the worst i have seen",  # neg
    "my new favorite movie because i love the main character",  # pos
    "watch this movie if you want to waste your time", # neg
    "this movie changed my life", # pos
    "i would like my money back", # neg
    "i have never been more inspired than i was after watching this" # pos
]
predicted_scores = export_model.predict(inputs)
predicted_labels = get_string_labels(predicted_scores)
for input, label in zip(inputs, predicted_labels):
  print("Question: ", input)
  print("Predicted label: ", label.numpy())


# PLOT PERFORMANCE 
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()