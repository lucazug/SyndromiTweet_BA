
'''''
#  --Bachelorarbeit Luca Zug--
#
#  --keyword_selecmodel.py
#  Extracting significant keywords and found words from keyword Twitter data. The significant keywords are determined by
#  their frequency of occurrence in the data and whether they are also listed in a separate "Symptome" CSV file. The 
#  significant found words are determined by their frequency of occurrence in the data and whether they occur more
#  frequently than the square of the median frequency of all found words.
#  
'''''

from typing import List, Any
import pandas as pd
from pandas import DataFrame
import numpy as np
import funcs

# Read in the csv file of Twitter data and drop any duplicates.
df = pd.read_csv('twitter_data/Twitter_Keyword_Selec_final.csv', sep=";")
df = df.drop_duplicates()

# Read in "Symptome" csv file
symptome = pd.read_csv('Symptome.csv')

# Keep only rows with a class value of 1, keep only the "text" column
print(len(df[df['class'] == 1]))
df = df[df['class'] == 1]
df = df['text']

# Remove emojis, URLs from text and clean text by lowercasing and removing special characters
df = df.apply(funcs.remove_emoji)
df = df.apply(funcs.remove_url)
df = df.apply(funcs.clean_text)

# Create a copy of the data with stop words included
df_with_stop = df
df = df.apply(funcs.stop_word_removal)

# Create an empty dictionary and iterate through each row of the data
d = dict()
for row in df:

    # Split row into a list of words
    words = row.split(" ")

    # Iterate through each word in the list
    for word in words:
        if word in d:
            # If the word is already in the dictionary, increment its value by 1
            d[word] = d[word] + 1
        else:
            # If the word is not in the dictionary, add it with a value of 1
            d[word] = 1

# A dictionary of the words in the dataframe that also appear in the list of pre-defined keywords
symp = symptome.set_index('QueryTerms').T.to_dict()
keywords = {k: d[k] for k in d if k in symp}
print(keywords)
print(d)

# Create a list of lists, each inner list containing a word and its frequency in the dataframe
keywords_list = []
for key, val in keywords.items():
    keywords_list.append([key, val])

# Create a dataframe from the keywords_list.
kwdf = pd.DataFrame.from_dict(keywords_list)

# A list of the frequencies of the keywords, mean_kw: the mean frequency of the keywords
num = list(keywords.values())
mean_kw = np.mean(num)
print(mean_kw)

# A dataframe of the keywords that have a frequency greater than the mean frequency
significant_kwdf = kwdf[kwdf[1] > mean_kw]
print(significant_kwdf)
significant_kwdf.loc[:, 0].to_csv("Significant_Keywords.csv")

# Create a list of lists, each inner list containing a word and its frequency in the dataframe
foundwords_list: list[list[int | Any]] = []
for key, val in d.items():
    foundwords_list.append([key, val])
fwdf = pd.DataFrame.from_dict(foundwords_list)

# Create a dataframe of the foundwords_list with only words that occur more than once
fwdf = fwdf[fwdf[1] > 1]

# The median frequency of the words in foundwords df, squared. Create a list of words with a frequency larger than this
# cutoff value. The cutoff value was chosen to exclude very low frequency words in a dataset, in which few words have
# an extremely high frequency and a lot of words have low frequency. Thus, a squared median returned the best results.
median_fw = np.median(fwdf[1])**2
print(median_fw)
significant_fwdf = fwdf[fwdf[1] > median_fw]
print(significant_fwdf)
significant_fwdf.loc[:,0].to_csv("Significant_Foundwords.csv")
