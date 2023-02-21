
'''''
#  --Bachelorarbeit Luca Zug--
#
#  --nlp_train_bert.py
#  Loading manually labelled data, preprocessing to remove emojis, upper case letters, punctuation etc., train-test-
#  validation split, get GBERT-base embeddings through the Huggingface API (HTTP requests), convert into tensor,
#  specify convolutional neural network, train and test the CNN. This files saves the CNN to be used in 
#  nlp_prediction_bert.py
#
#  Please install the necessary packages & dependencies by running "pip install -r requirements.txt" in the Terminal.
#
'''''

import time
import requests
import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
from tensorflow.keras import regularizers

from tensorflow.keras import layers
from tensorflow.keras import losses
from keras.utils.vis_utils import plot_model

from collections import Counter

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight
from sklearn.metrics import roc_curve


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.metrics import AUC
import json
import pydot
import funcs
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pickle
import bz2
import _pickle as cPickle

## ++ DATA PROCESSING ++ ##

# Read in main data from CSV file, removing rows with missing values in the 'class' column
main_data = pd.read_csv("twitter_data/230103_Twitter_manual_coding_finished.csv", sep=";")
main_data.dropna(axis=0, how='any', inplace=True, subset=['class'])

# Remove duplicate rows based on the 'text' column
main_data.drop_duplicates(subset=['text'], keep="first", inplace=True)

# Print number of samples for each class and total number of samples
print('-------Train data--------')
print(main_data['class'].value_counts())
print(len(main_data))
print('-------------------------')

# Apply the following functions to the 'text' column in order to clean it:
#   - remove_name: removes names from text
#   - remove_emoji: removes emojis from text
#   - remove_url: removes URLs from text
#   - clean_text: removes punctuation, lowercases text, and removes extra whitespace
main_data['text'] = main_data['text'].apply(funcs.remove_emoji)
main_data['text'] = main_data['text'].apply(funcs.remove_url)
main_data['text'] = main_data['text'].apply(funcs.clean_text)
main_data['class'] = main_data['class'].apply(int)
main_data.dropna(axis=0, how='any', inplace=True, subset=['text'])

# Only keep rows where the number of words in the 'text' column is greater than two
main_data['Num_words_text'] = main_data['text'].apply(lambda x: len(str(x).split()))
print(main_data['Num_words_text'])
mask = main_data['Num_words_text'] > 2
main_data = main_data[mask]

# Split main_data into training, validation, and test sets
X = main_data.drop(['class'], axis=1)
y = main_data['class']

X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.2, random_state=33)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.2, random_state=33)

# Calculate maximum sentence length for training and test sets
max_train_sentence_length = X_train['Num_words_text'].max()
max_test_sentence_length = X_test['Num_words_text'].max()

print('X Train Max Sentence Length :' + str(max_train_sentence_length))
print('X Test Max Sentence Length :' + str(max_test_sentence_length))

# Save data for later usage in the model evaluation
X_train_bu = X_train
X_train_ntd = X_train.drop(['text'], axis=1)
X_valid_ntd = X_valid.drop(['text'], axis=1)
X_test_ntd = X_test.drop(['text'], axis=1)
X_valid_bu = X_valid
X_test_bu = X_test
X_test_non_token = np.array(X_test['text'])
X_test_non_token_df = X_test['text']
y_test_bu = y_test

# The following section was used to obtain and save-to-file GBERT embeddings for training, valid and test datasets.
# The code is commented out because the embeddings have already been obtained by the author. They are loaded and
# prepared for the model from line 190 to 202.
'''
# Convert 'text' column in training, validation, and test sets to lists
X_train = X_train['text'].tolist()
X_valid = X_valid['text'].tolist()
X_test = X_test['text'].tolist()

# Convert 'class' column in training, validation, and test sets to numpy arrays
y_train = np.asarray(y_train)
y_valid = np.asarray(y_valid)
y_test = np.asarray(y_test)

# Get BERT embeddings for training, validation, and test sets through huggingface API
inputs_train, lost_train, y_train, X_train_bu = funcs.get_bert_embeddings(X_train, y_train, X_train_bu)
print('X_train finished')

with open('lost_train.json', 'w') as f:
  json.dump(lost_train, f)

inputs_valid, lost_valid, y_valid, X_valid_bu = funcs.get_bert_embeddings(X_valid, y_valid, X_valid_bu)
print('X_valid finished')

with open('lost_valid.json', 'w') as f:
  json.dump(lost_valid, f)

inputs_test, lost_test, y_test, X_test_bu = funcs.get_bert_embeddings(X_test, y_test, X_test_bu)
print('X_test finished')

with open('lost_test.json', 'w') as f:
  json.dump(lost_test, f)

# Find the longest list in the training, validation, and test sets
max_len_train = funcs.find_longest_list(inputs_train)
max_len_valid = funcs.find_longest_list(inputs_valid)
max_len_test = funcs.find_longest_list(inputs_test)

# Set 'max_len' to the longest list length + 10
max_len = max(max_len_train, max_len_valid, max_len_test)
max_len = max_len + 10
print(max_len)

# Pad list of lists to same length (max_len) to convert into tensor . Save word embeddings as pickle files.
inputs_train = funcs.manual_padding(inputs_train, 768, max_len)
funcs.compressed_pickle('inputs_train_embeddings', inputs_train)

json_train = json.dumps(inputs_train)
with bz2.open('train.json.bz2', 'wb') as f:
  f.write(json_train.encode())

inputs_valid = funcs.manual_padding(inputs_valid, 768, max_len)
funcs.compressed_pickle('inputs_valid_embeddings', inputs_valid)

json_valid = json.dumps(inputs_valid)
with bz2.open('valid.json.bz2', 'wb') as f:
  f.write(json_valid.encode())

inputs_test = funcs.manual_padding(inputs_test, 768, max_len)
funcs.compressed_pickle('inputs_test_embeddings', inputs_test)

json_test = json.dumps(inputs_test)
with bz2.open('test.json.bz2', 'wb') as f:
  f.write(json_test.encode())

# Save numpy arrays of class data to file
np.save('array_data/y_train_array', y_train)
np.save('array_data/y_valid_array', y_valid)
np.save('array_data/y_test_array', y_test)
'''

y_train = np.load('array_data/y_train_array.npy', allow_pickle=True)
y_test = np.load('array_data/y_test_array.npy', allow_pickle=True)
y_valid = np.load('array_data/y_valid_array.npy', allow_pickle=True)

inputs_train = funcs.decompress_pickle("inputs_train_embeddings.pbz2")
inputs_valid = funcs.decompress_pickle("inputs_valid_embeddings.pbz2")
inputs_test = funcs.decompress_pickle("inputs_test_embeddings.pbz2")
print("All objects have been loaded. Let's start converting to tensors…")

inputs_train_tensor = tf.convert_to_tensor(inputs_train, dtype=tf.float64)
inputs_valid_tensor = tf.convert_to_tensor(inputs_valid, dtype=tf.float64)
inputs_test_tensor = tf.convert_to_tensor(inputs_test, dtype=tf.float64)
print("Tensorised! Let's actually begin…")

### Convolutional Neural Network

## Define model parameters

# Activation Functions for Convolutional Layers: ReLU
# Acivation Function for Output Layer (Binary Classes): Sigmoid
# Loss Function: Binary Crossentropy
# Optimizer: NAdam

optm = tf.keras.optimizers.SGD(learning_rate=3e-4)

model_cnn = tf.keras.Sequential()
model_cnn.add(tf.keras.layers.Conv1D(filters=128, kernel_size=2, strides=3, activation='relu', padding='same',
                                     kernel_regularizer=regularizers.l2(0.0005),
                                     bias_regularizer=regularizers.l2(0.03)))
model_cnn.add(tf.keras.layers.Conv1D(filters=128, kernel_size=4, strides=3, activation='relu', padding='same',
                                     kernel_regularizer=regularizers.l2(0.0005),
                                     bias_regularizer=regularizers.l2(0.03)))
model_cnn.add(tf.keras.layers.GlobalMaxPooling1D())
model_cnn.add(tf.keras.layers.Dropout(0.5))
model_cnn.add(tf.keras.layers.Dense(64, activation='relu', bias_regularizer=regularizers.l2(0.0001)))
model_cnn.add(tf.keras.layers.Dense(32, activation='relu', bias_regularizer=regularizers.l2(0.0001)))
model_cnn.add(tf.keras.layers.Dense(1, activation='sigmoid', bias_regularizer=regularizers.l2(0.003)))
model_cnn.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer=optm,
                      metrics=["Accuracy", "Precision", "Recall"])

epochs = 90

# The class labels are highly imbalanced, which impacts model performance. The weights to balance the labels are
# calculated and applied in model training.
class_weights = class_weight.compute_class_weight(classes=np.unique(y_train), class_weight='balanced', y=y_train)
classes = np.unique(y_train)
class_weights = dict([(int(classes[0]), class_weights[0]), (int(classes[1]), class_weights[1])])

# The model is trained with a batch size of 128, balanced class weights and validation data.
history = model_cnn.fit(inputs_train_tensor, y_train, epochs=epochs, verbose=1, batch_size=128,
                        class_weight=class_weights, validation_data=(inputs_valid_tensor, y_valid))

# The model is saved to the harddrive for later retrieval for prediction.
#model_cnn.save('tf_cnnmodel')

# Plot the accuracy and precision of the model over all training epochs.
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy and Precision')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()

# Load the pretrained model into python.
model_cnn = tf.keras.models.load_model('tf_cnnmodel')
print("Pretrained CNN is loaded.")

# Evaluating model performance on the validation set data with
print(model_cnn.evaluate(inputs_valid_tensor, y_valid))
y_pred_valid = model_cnn.predict(inputs_valid_tensor)
cm_cnn_valid = confusion_matrix(y_valid, np.around(y_pred_valid))
print(cm_cnn_valid)

# *TESTING*
# As random_state = 33 has been set in the train-test-valid split in line 96 an 97, the data splits remain the same
# as in the word embeddings (same random state).

y_pred_test = model_cnn.predict(inputs_test_tensor)
y_pred_test_round = pd.DataFrame(np.round(y_pred_test), index=y_test_bu.index, columns=['Predicted Values'])
predictions_cnn = y_pred_test_round

print(model_cnn.evaluate(inputs_test_tensor, y_test))

# Print and plot a confusion matrix of the test data
cm_cnn_test = confusion_matrix(y_test, np.around(y_pred_test), labels=[0, 1])
print(cm_cnn_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_cnn_test, display_labels=[0, 1])
disp.plot()
plt.show()

# Calculate AUC
auc_roc = AUC(name='auc_roc')
auc_roc.update_state(y_test, y_pred_test)
print("AUC: ", auc_roc.result().numpy())

# Get ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)

# Plot ROC curve
plt.plot(fpr, tpr, 'b-', label='AUC = %0.2f' % auc_roc.result().numpy())
plt.plot([0, 1],[0, 1],'r--')
plt.xlim([0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.ylim([0, 1])
plt.title("LOC Curve for CNN")
plt.grid()
plt.legend(loc='lower right')
plt.show()

# Save predictions and actual values in a csv-file
y_test_df = pd.DataFrame(y_test_bu, columns=['True Labels'], index=y_test_bu.index)
predictions_cnn = pd.concat([X_test_bu, predictions_cnn, y_test_df], ignore_index=False, axis=1)
predictions_cnn.to_csv(path_or_buf='predicted_Tweets.csv', index=True, header=True)
