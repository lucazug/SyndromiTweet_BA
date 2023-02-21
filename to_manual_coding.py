'''''
#  --Bachelorarbeit Luca Zug--
#
#  --to_manual_coding.py
#  Load main Twitter data, pull indices from truncated shifted normal distribution and only include these indices when
#  writing the file to csv for manual labelling.
#
'''''

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats as stats

df = pd.read_csv('twitter_data/Twitter_main_final_fixed.csv', dtype={'author_id': 'str',
                                                                     'username': 'str',
                                                                     'author_followers': 'int',
                                                                     'author_location': 'str',
                                                                     'text': 'str',
                                                                     'created_at': 'str',
                                                                     'tweet_location': 'str'})

selectlist = []

# Generate a list of random indices from the dataset using a truncated normal distribution

# Lower and upper bounds for the distribution
lower, upper = 0, len(df)

# Mean and standard deviation for the distribution – 47500 represents the date of the approx. peak of the influenza wave
# 2019 to obtain more relevant tweets.
mu, sig = 47500, 90000
truncatednorm = stats.truncnorm((lower - mu) / sig, (upper - mu) / sig, loc=mu, scale=sig)

# Generate random indices from the distribution -- Round the indices to the nearest integer
selectlist = np.around(truncatednorm.rvs(10000))
print(selectlist)

# Plot a histogram of the generated list of indices
plt.hist(selectlist, bins=2000)
plt.title('Histogram: Distribution of Amount of Selected Tweets for Coding')
plt.show()

# Select rows from the dataset based on the generated list of indices
df_manual = df.loc[selectlist]
df_manual.to_csv('twitter_data/221220_Twitter_to_manual_coding.csv')
