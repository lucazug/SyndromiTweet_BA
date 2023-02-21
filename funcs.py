
'''''
#  --Bachelorarbeit Luca Zug--
#
#  --funcs.py
#  Python file with background functions for other files:
#       compressed_pickle:      Compresses and saves data to a .pbz2 file.
#       decompress_pickle:      Decompresses and loads data from a .pbz2 file.
#       remove_name:            Removes names from an input text (tweet).
#       remove_emoji:           Removes emojis from text.
#       remove_url:             Removes URLs from text.
#       clean_text:             Cleans text by removing punctuation, lowercasing, and removing extra whitespace.
#       stop_word_removal:      Removes stop words from nltk library (corpus) text.
#       word_freq:              Calculates frequency of each word in a string.
#       random_time_interval:   Generates random time intervals within a given range.
#       random_time_of_day:     Generates random time intervals for a given number of days.
#       shortenby:              This function takes a length and a dataframe as input, converts the dataframe to a list, 
                                truncates each element of the list to the given length, and returns the modified list.
#       cal_week_for_date:      This function takes a dataframe as input, extracts the date from each element of the 
                                first column of the dataframe, calculates the week of the year for each date, adds the 
                                weeks as a new column to the dataframe, and returns the modified dataframe.
#       manual_padding:         This function takes a list of lists, an embedding dimension, and 
                                a maximum length as input, pads the inner lists with lists of 
                                zeros so that all inner lists have the same length, and returns the padded input.
#       random_time_of_day:     This function generates two lists of random integers representing hours of the day, with 
                                one list corresponding to the start time and the other to the end time. It then returns 
                                these lists.
#       find_longest_list:      Finds the length of the longest list in a list of lists
#       get_bert_embeddings:    Sends a request to a server to get BERT embeddings for a list of texts and returns the 
                                embeddings in a list  
#
'''''

import os
import re
import shutil
import string
#import url
import numpy as np
from nltk.corpus import stopwords
import random
from datetime import datetime, timedelta, date
import time
import pandas as pd
import math
from sklearn.feature_extraction.text import CountVectorizer
import itertools
import scipy.stats as stats
import requests
import pickle
import _pickle as cPickle
import bz2

def compressed_pickle(title, data):
    """
    Function to compress and save data to a .pbz2 file.
    :param title:
    :param data:
    :return: / # Save file to hard drive.
    """
    with bz2.BZ2File(title + '.pbz2', 'w') as f:
        cPickle.dump(data, f)

def decompress_pickle(file):
    """
    Function to decompress and load data from a .pbz2 file.
    :param file:
    :return: data # Unzipped file.
    """

    data = bz2.BZ2File(file, "rb")
    data = cPickle.load(data)
    return data

def remove_name(tweet):
    """
    Function to remove Twtter handles from an input text (tweet) if there is an @ before the handle.

    :param tweet:
    :return: # Tweet without @ Twitter handles, that is, list of words.
    """

    words = tweet.split()
    for i, word in enumerate(words):
        if word[0] == '@':
            del words[i]
    modified_tweet = ' '.join(words)
    return modified_tweet

def remove_emoji(text):
    """
    Function to remove emojis from text.
    :param text:
    :return:
    """

    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_url(text):
    """
    Function to remove URLs from text.

    :param text:
    :return: # List of words without any URLs.
    """

    url_pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.sub(r'', text)

def clean_text(text):
    """
    Function to clean text by removing punctuation, lowercasing, and removing extra whitespace.

    :param text:
    :return: #Cleaned list of words.
    """

    delete_dict = {sp_character: '' for sp_character in string.punctuation}
    delete_dict[' '] = ' '
    table = str.maketrans(delete_dict)
    text1 = text.translate(table)
    textArr = text1.split()
    text2 = ' '.join([w for w in textArr if (not w.isdigit() and (not w.isdigit() and len(w) > 2))])

    return text2.lower()

# Load German stop words from NLTK package
german_stop_words = stopwords.words('german')
vect = CountVectorizer(stop_words = german_stop_words)

def stop_word_removal(x):
    """
    Function to remove stop words from text.

    :param x:
    :return: #List of tokens without stopwords
    """

    token = x.split()
    return ' '.join([w for w in token if not w in german_stop_words])

def word_freq(string, baselist):
    """
    Function to calculate frequency of each word in a string.

    :param string:
    :param baselist:
    :return: baselist
    """
    # Split string into list of words
    wordlist = string.split()
    print(wordlist)

    # Loop through wordlist and update frequency of word in baselist
    for i in wordlist:
        for word in baselist:
            if word in i:
                baselist[word,"freq"] +=1
    # Return updated baselist
    return baselist

def random_time_interval(number, year, start_day, start_month, end_day, end_month):
    """
    Function to generate random time intervals within a given range.
    :param number:
    :param year:
    :param start_day:
    :param start_month:
    :param end_day:
    :param end_month:
    :return: dateframe # List of random time intervals in a time range.
    """

    # Convert start and end dates to timestamps
    start_stamp = datetime(year, start_month, start_day, 0, 0).timestamp()
    end_stamp = datetime(year, end_month, end_day, 0, 0).timestamp()

    # Generate list of random integers between start and end timestamps
    randomlist = random.sample(range(int(start_stamp), int(end_stamp)), number)

    # Convert random timestamps to ISO formatted strings
    datelist = list()
    for n in range(0, number):
        temp = datetime.fromtimestamp(randomlist[n]).isoformat()
        datelist.append(str(temp))

        # Add 'Z' to end of date string
        datelist[n] = datelist[n] + 'Z'

    # Create end date strings by replacing time portion with '23:59:59Z'
    datelist_end = list()
    for n in range(0, number):
        datelist_end.append(str(datelist[n]))
        datelist_end[n] = datelist_end[n][:11] + '23:59:59Z'

    # Create pandas DataFrame with start and end dates
    dateframe = pd.DataFrame(datelist, columns=['start_date'])
    dateframe['end_date'] = datelist_end
    return dateframe

def random_time_of_day():
    """
    Function to generate time of day from 00:00:00 am until 23:59:59 pm for every day within a given year.
    :return: start_time, end_time
    """

    # Create a range of dates for the given year
    start_stamp = pd.date_range(start='2019-01-01', end='2019-12-31')

    # Convert dates to ISO formatted strings with time set to '00:00:00Z'
    start_time = list()
    for n in range(0, len(start_stamp)):
        start_time.append(str(start_stamp[n]))
        start_time[n] = str(start_time[n][:10]) + 'T' + '00:00:00Z'

    # Convert dates to ISO formatted strings with time set to '23:59:59Z'
    end_time = list()
    for n in range(0, len(start_stamp)):
        end_time.append(str(start_stamp[n]))
        end_time[n] = str(end_time[n][:10]) + 'T' + '23:59:59Z'

    # Return start and end time lists
    return (start_time, end_time)

# Below code is the original code for the function random_time_of_day(). Original code returned truncated Gaussian
# distributed times of day to reduce the number of tweets obtained for every day. However, testing showed that
# this was not necessary as the file sizes were not excessive.
'''
    start_stamp = pd.date_range(start='2019-12-01', end='2019-12-30')
    lower, upper = 0, 23
    mu_morning, sigma_morning = 9, 9
    mu_evening, sigma_evening = 18, 11
    truncatednorm_morn = stats.truncnorm((lower - mu_morning) / sigma_morning, (upper - mu_morning) / sigma_morning, loc=mu_morning, scale=sigma_morning)
    truncatednorm_even = stats.truncnorm((lower - mu_evening) / sigma_evening, (upper - mu_evening) / sigma_evening, loc=mu_evening, scale=sigma_evening)
    randomlist = [truncatednorm_morn.rvs() for p in range(0, 365)]
    secondrandomlist = [truncatednorm_even.rvs() for p in range(0, 365)]

    randomlist = [round(p) for p in randomlist]
    secondrandomlist = [round(p) for p in secondrandomlist]

    for i in range(0, len(start_stamp)):
        if randomlist[i] > secondrandomlist[i]:
            if randomlist[i] == 23:
                randomlist[i] = randomlist[i] - 1
            singlerandom = random.sample(range(randomlist[i]+1,24), 1)
            secondrandomlist[i] = singlerandom[0]
        else: pass
        if randomlist[i] == secondrandomlist[i]:
            temp_rand_morn = truncatednorm_morn.rvs(1)
            randomlist[i] = round(temp_rand_morn[0])
            temp_rand_even = truncatednorm_even.rvs(1)
            secondrandomlist[i] = round(temp_rand_even[0])
        else: pass
        if randomlist[i] > secondrandomlist[i]:
            if randomlist[i] == 23:
                randomlist[i] = randomlist[i] - 1
        singlerandom = random.sample(range(randomlist[i] + 1, 24), 1)
        secondrandomlist[i] = singlerandom[0]

    zerolist = list()
    for n in range(0, len(randomlist)):
        if randomlist[n] < 10:
            zerolist.append(str(randomlist[n]).zfill(2))
        else:
            zerolist.append(str(randomlist[n]))

    start_time = list()
    for n in range(0, len(start_stamp)):
        start_time.append(str(start_stamp[n]))
        start_time[n] = str(start_time[n][:10]) + 'T' + zerolist[n] + ':00:00Z'

    s_zerolist = list()
    for n in range(0, len(secondrandomlist)):
        if secondrandomlist[n] < 10:
            s_zerolist.append(str(secondrandomlist[n]).zfill(2))
        else:
            s_zerolist.append(str(secondrandomlist[n]))

    end_time = list()
    for n in range(0, len(start_stamp)):
        end_time.append(str(start_stamp[n]))
        end_time[n] = str(end_time[n][:10]) + 'T' + s_zerolist[n] + ':00:00Z'

    return(start_time, end_time)
'''

def shortenby(length, df):
    """
    Function to shorten the dates in the given dataframe by the given length of characters. The shortened dates are
    returned in a list.
    :param length:
    :param df:
    :return temporary (list of strings):
    """
    temporary = list()
    for n in range(0, len(df)):
        temporary.append(str(df[n]))
        temporary[n] = str(temporary[n][:length])
    return temporary

def cal_week_for_date(df):
    """
    Function to calculate the week of the year for each date in the given dataframe. The weeks are returned in a list
    and added as a new column to the dataframe.
    :param df:
    :return: df
    """
    temp = list()
    year = list()
    month = list()
    day = list()
    weeklist = list()

    tempdf = df[0]
    for n in range(0, len(df)):
        temp.append(tempdf[n])
        year = int(temp[n][:4])
        month = int(temp[n][5:7])
        day = int(temp[n][8:])
        weeklist.append(date(year, month, day).isocalendar()[1])
    df['calweek'] = weeklist
    return df


def manual_padding(input, embedding_dim, max_len):
    """
    Function to pad the given input list of lists with inner lists of zeros, so that all inner lists have the same length.
    The length is specified by the max_len parameter. The padded input is returned.
    :param input:
    :param embedding_dim:
    :param max_len:
    :return: input
    """

    for n in range(0, len(input)):
        needed_len = max_len - len(input[n])
        for j in range(needed_len):
            innermost_list = [0] * embedding_dim
            input[n].append(innermost_list)
    return input

def find_longest_list(lists):
    """
    Function to find the length of the longest list in a list of lists (@lists: list of lists).
    Returns the length of the longest list in the list of lists.
    :param lists:
    :return longest_length:
    """

    longest_length = 0
    for lst in lists:
        if len(lst) > longest_length:
            longest_length = len(lst)
    return longest_length

def get_bert_embeddings(input, y_input, X_bu):
    """
    This function uses the Hugging Face API to get BERT embeddings for each text in the input list.
    For loops are used because the server only accepts single lines for HTTP requests.
    A websocket could not be implemented due to the lack of a corresponding URI for the Inference API GBERT and cloning
    the model on a local machine was unsuccessful so individual requests must be used.
    :param input:
    :param y_input:
    :param X_bu:
    :return output, lostlist, y_input, X_bu:
    """

    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/deepset/gbert-base"
    headers = {"Authorization": f"Bearer ## HIDDEN ##"}
    counter = 0
    output = []
    lostlist = []

    # Iterate through each text in the input list
    for text in input:
        try:
            print(text)
            print(f"{counter} von {len(input)}")
            counter += 1

            # Send an HTTP request to the API to get the BERT embeddings for the current text
            response = requests.post(api_url, headers=headers, json={"inputs": text, "options":{"wait_for_model": True}})

            # Get the embeddings from the response and append to list
            embeddings = response.json()[0]
            output.append(embeddings)

        # If there is a KeyError (too many requests or other error), wait for 20 seconds and retry same as above
        except KeyError as e:
            print('To many requests or other error, sleeping for 10 Sec and retrying.')
            print(e)
            time.sleep(20)
            try:
                print(text)
                print(f"{counter} von {len(input)}")
                response = requests.post(api_url, headers=headers, json={"inputs": text, "options": {"wait_for_model": True}})
                embeddings = response.json()[0]
                output.append(embeddings)

            # If there is another KeyError (too many requests or other error), wait for 20 seconds and retry same
            # as above
            except KeyError:
                print('To many requests or other error, sleeping for 10 Sec and retrying.')
                time.sleep(20)
                try:
                    print(text)
                    print(f"{counter} von {len(input)}")
                    response = requests.post(api_url, headers=headers,
                                             json={"inputs": text, "options": {"wait_for_model": True}})
                    embeddings = response.json()[0]  # ['features'][0]['layers'][0]['values']
                    output.append(embeddings)

                # If the error persists, delete the current item and move on to the next one.
                except KeyError:
                    print("Delete number", counter, "in y")
                    y_input = np.delete(y_input, counter)
                    y_input = y_input.astype(int)

                    print("Delete the following text from X_bu", text)
                    X_bu.drop(X_bu[X_bu['text'] == text].index, inplace=True)

                    lostlist.append(text)
                    print(lostlist)
                    pass

        # If a ConnectionError is raised (machine is not connected to the internet), wait for 60 seconds and
        # try to reconnect.
        except requests.ConnectionError:
            print("The machine is not connected to the internet")
            time.sleep(60)
            try:
                response = requests.get("https://www.google.com")
                print("Successfully reconnected to the internet")

            # If there is still a ConnectionError after attempting to reconnect, print that the machine failed to
            # reconnect to the internet.
            except requests.ConnectionError:
                print("Failed to reconnect to the internet")

        # If a JSONDecodeError is raised, retry the request. If the error persists, delete the current item and move on
        # to the next one.
        except requests.exceptions.JSONDecodeError:
            print("Oh oh, JSON decoding error")
            try:
                print(text)
                print(f"{counter} von {len(input)}")
                response = requests.post(api_url, headers=headers, json={"inputs": text, "options": {"wait_for_model": True}})
                embeddings = response.json()[0]
                output.append(embeddings)
            except requests.exceptions.JSONDecodeError:

                print("Delete number", counter)
                y_input = np.delete(y_input, counter)
                y_input = y_input.astype(int)

                print("Delete the following text from X_bu", text)
                X_bu.drop(X_bu[X_bu['text'] == text].index, inplace=True)

                lostlist.append(text)
                print(lostlist)
                pass

    # The output of this function is a list of BERT embeddings, where each embedding is a 768-length vector representing
    # the BERT embedding for a word in the input text. This list will have the same length as the input list, with the
    # i-th element being the BERT embeddings for the i-th text in the input list (n(i) x 768).
    return output, lostlist, y_input, X_bu

def normalise(df, col1, col2):
    """
    Take a dataframe and min-max-normalise two predefined columns of that dataframe.
    :param df:
    :param col1:
    :param col2:
    :return new_df: #Dataframe with two normalised columns.
    """
    new_df = df
    columns_to_normalise = [col1, col2]

    # Min-max normalize the columns
    new_df[columns_to_normalise] = (new_df[columns_to_normalise] - new_df[columns_to_normalise].min()) \
                                   / (new_df[columns_to_normalise].max() - new_df[columns_to_normalise].min())
    return new_df

def replaceNaN(df, column):
    """
    Replace NaN values with the average of the value above and below the row.
    :param df(pandas.DataFrame):
    :param column:
    :return df(pandas.DataFrame):
    """
    col = df[column]

    for i in range(1, len(df) - 1):
        if np.isnan(col[i]):
            col[i] = (col[i-1] + col[i+1]) / 2
    df[column] = col
    return df
