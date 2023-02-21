
'''''
#  --Bachelorarbeit Luca Zug--
#
#  --nlp_train_prediction.py
#  Loading manually labelled data, preprocessing to remove emojis, upper case letters, punctuation etc., train-test-
#  validation split, get GBERT-base embeddings through the Huggingface API (HTTP requests), convert into tensor,
#  specify convolutional neural network, 
#
#  Please install the necessary packages by running "pip install -r requirements.txt" in the Terminal.
#
'''''

import datetime
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
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from sklearn.pipeline import Pipeline

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
# Same preprocessing steps are undertaken as for the training data in 'nlp_train_bert.py'.

max_len = 148
model_cnn = tf.keras.models.load_model('tf_cnnmodel')

predict_data = pd.read_csv('twitter_data/Twitter_main_final_fixed.csv', dtype={'author_id': 'str',
                                                                     'username': 'str',
                                                                     'author_followers': 'int',
                                                                     'author_location':'str',
                                                                     'text':'str',
                                                                     'created_at': 'str',
                                                                     'tweet_location':'str'})

predict_data.dropna(axis=0, how='any', inplace=True, subset=['text'])
predict_data.drop_duplicates(subset=['text'], keep="first", inplace=True)

predict_data['text'] = predict_data['text'].apply(funcs.remove_name)
predict_data['text'] = predict_data['text'].apply(funcs.remove_emoji)
predict_data['text'] = predict_data['text'].apply(funcs.remove_url)
predict_data['text'] = predict_data['text'].apply(funcs.clean_text)

predict_data['Num_words_text'] = predict_data['text'].apply(lambda x: len(str(x).split()))
print(predict_data['Num_words_text'])
mask = predict_data['Num_words_text'] > 2
predict_data = predict_data[mask]

max_train_sentence_length = predict_data['Num_words_text'].max()
print('X Train Max Sentence Length :' + str(max_train_sentence_length))

# Create a list of length of main dataframe to feed into embedding retrieval function to fit function input variables.
y_fake = [0]*int(predict_data.shape[0])

def embed_predict(dataframe, y_fake, predict_data, x):
    """
    The dataframe for predicting all tweets returned through the Twitter API is split up as large amounts of data are
    returned which may not be processed by some machines. I define a pipeline, which takes the preprocessed dataframe
    as input and returns an array of predictions.

    :param dataframe:
    :param y_fake:
    :return: y_predict #list of predicitions
    """

    num = x
    inputs_predict = []
    inputs_predict, lostpredict, y_fake, predict_data = funcs.get_bert_embeddings(dataframe, y_fake, predict_data)
    with open(f'{num}_lost_predict.json', 'w') as f:
        json.dump(lostpredict, f)

    inputs_predict = funcs.manual_padding(inputs_predict, 768, max_len)

    funcs.compressed_pickle(f"_maincall_embeddings/{num}_embeddings_main_call", inputs_predict)

    inputs_predict_tensor = tf.convert_to_tensor(inputs_predict, dtype=tf.float64)
    y_predict = model_cnn.predict(inputs_predict_tensor)
    return y_predict, predict_data


def split_and_save(dataframe):
    """
    Defines a function named split_and_save which takes in a dataframe as an input. It splits the dataframe into smaller
    dataframes of around 20000 rows each by dividing the number of rows by 20000. Finally, it returns a list of the
    split dataframes.

    :param dataframe:
    :return: dfs # List of split dataframes.
    """

    num_parts = round(int(dataframe.shape[0]//20000))
    rows_per_part = dataframe.shape[0] // num_parts

    # Create a list to store the split dataframes
    dfs = []
    # Split the dataframe into number of parts
    for i in range(num_parts):
        start = i * rows_per_part
        # Check if this is the last part
        if i == num_parts - 1:
            # If it is, set the end index to the last row of the dataframe
            end = dataframe.shape[0]
        else:
            end = (i + 1) * rows_per_part
        dfs.append(dataframe[start:end])
    return dfs

def retrieve_predict(x):
    """
    The dataframe for predicting all tweets returned through the Twitter API is split up as large amounts of data are
    returned which may not be processed by some machines. I define a pipeline, which takes the preprocessed dataframe
    as input and returns an array of predictions.

    :param dataframe:
    :param y_fake:
    :return: y_predict #list of predicitions
    """
    num = x
    inputs_predict = funcs.decompress_pickle(f"_maincall_embeddings/{num}_embeddings_main_call.pbz2")
    inputs_predict_tensor = tf.convert_to_tensor(inputs_predict, dtype=tf.float64)
    y_predict = model_cnn.predict(inputs_predict_tensor)
    return y_predict

split_dfs = split_and_save(predict_data)

y_predict = []

# Loop through each dataframe in the list 'split_dfs', list of dataframes that contain the entire original
# ca. 290.000 rows.
for i in range(0, len(split_dfs)):

    # Use the 'embed_predict()' function to get the predicted values and update 'predict_data'
    y_temp, predict_data = embed_predict(split_dfs[i]['text'].tolist(), y_fake, predict_data, i)

    # Append the predicted values to the 'y_predict' list
    y_predict.append(y_temp)
    print("+++ One more list done +++")

# The following code is used in case the embeddings for the main dataframe have already been obtained and saved to the
# hard drive.
'''
# Load 'lost_predict.json' file containing all tweets that caused an error during embedding retrieval
with open('lost_predict.json', 'r') as f:
    lostpredict = json.load(f)

# Remove the 'lostpredict' data from 'predict_data'
print(predict_data.shape)
predict_data = predict_data[~predict_data['text'].isin(lostpredict)]
print(predict_data.shape)

# Randomly sample 'predict_data' and remove 255 rows that were lost during embeddings retrieval but were not captured in 
# 'lostpredict'. This represents only 0,08% of the dataset.
predict_data = predict_data.sample(len(predict_data)-255)

# Loop through each dataframe in the list 'split_dfs'
for i in range(0, len(split_dfs)):
    # Use the 'retrieve_predict()' function to get the predicted values from the saved word embedding files.
    y_predict.append(retrieve_predict(i))
    print(f"+++ List {i} of {len(split_dfs)-1} done +++")
'''

# Create a dataframe from the concatenated predicted values and the original 'predict_data'
predictions_cnn = pd.DataFrame(np.concatenate(y_predict, axis=0), index=predict_data.index, columns=['Predicted Values'])
predictions_cnn.to_csv(path_or_buf='predictions_only.csv', index=True, header=True)

predictions_cnn = pd.concat([predictions_cnn, predict_data], ignore_index=False, axis=1)
predictions_cnn.to_csv(path_or_buf='predicted_Tweets_MAIN.csv', index=True, header=True)
