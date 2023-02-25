'''''
#  --Bachelorarbeit Luca Zug--
#
#  --pull_tweets.py
#  Read list of preselected keywords, create Twitter API search query and call tweets through the API, convert result to 
#  DataFrame, remove tweets with @marker-accounts, and write csv-file with main data.
#
'''''

import tweepy as twp
import configparser
import requests
import time
import pandas as pd
import json
import funcs

# Read significant keywords from csv file and create search query by joining it with boolean query operators, returns
# a single string.
tweet_query = pd.read_csv("Significant_Keywords.csv", index_col=[0], sep=";")
x = tweet_query.to_string(header=False, index=False).split('\n')
x = [' '.join(ele.split()) for ele in x]
querylist = [' -is:retweet -has:media OR '.join(x)]
tweet_query = querylist[0] + ' -is:retweet -has:media'
print(tweet_query)

# Define bearer token and tweepy client with wait on rate limit to prevent timeouts
#bearer_token =  ## **REMOVED** ##
client = twp.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

ili_tweets = []
result = []
user_dict = {}
tweet_list = list()

# Generate a list of dates for every day of 2019 using 'random_time_interval' from python file for background
# functions (funcs). First, random time frames within all days of 2019 were tried. However, the file size for
# all hours of the day of every day of 2019 turned out to be handleable. Thus, all hours of every day of 2019 were used
# in the API call.
start_time, end_time = funcs.random_time_of_day()

# Iterate over start and end times and search for tweets in each interval.
for t in range(0,len(start_time)):
    start = start_time[t]
    end = end_time[t]

    # Search tweets in the current interval and append to ili_tweets
    print(start, end)
    for response in twp.Paginator(client.search_all_tweets,
                                    query = tweet_query,
                                    user_fields = ['username', 'public_metrics', 'description', 'location'],
                                    tweet_fields = ['created_at', 'geo', 'text'],
                                    expansions = 'author_id',
                                    start_time = start,
                                    end_time = end,
                                    max_results=500):
            time.sleep(1)
            ili_tweets.append(response)

final_result = []
user_dict = {}

# Iterate over the list 'ili_tweets' and extract relevant information about the user and tweet. This information is
# stored in a list of dictionaries called 'result'.
for response in ili_tweets:
    for user in response.includes['users']:
        user_dict[user.id] = {'username': user.username,
                              'followers': user.public_metrics['followers_count'],
                              'tweets': user.public_metrics['tweet_count'],
                              'description': user.description,
                              'location': user.location
                              }
    for tweet in response.data:
        author_info = user_dict[tweet.author_id]
        final_result.append({'author_id': tweet.author_id,
                       'username': author_info['username'],
                       'author_followers': author_info['followers'],
                       'author_tweets': author_info['tweets'],
                       'author_description': author_info['description'],
                       'author_location': author_info['location'],
                       'text': tweet.text,
                       'created_at': tweet.created_at,
                       'tweet_location': tweet.geo
                       })

df = pd.DataFrame(final_result)

# Remove tweets that mention specific accounts that most certainly will not be relevant to the classification task,
# that is, mention only political topics or news related topics. Please refer to the thesis for a detailed explanation.
print(df.shape)
droplist = ['@spdde', '@AfD', '@larsklingbeil', '@SPIEGELONLINE', '@realDonaldTrump', '@faznet', '@politico', '@fdp',
            '@KuehniKev', '@sigmargabriel', '@SerapGueler', '@ZDFheute', '@mschlapp', '@LeFloid', '@hubertus_heil',
            '@Tagesspiegel', '@BILD', '@jensspahn', '@c_lindner', '@Kachelmann', '@MontanaBlack', '@steam_games',
            '@Beatrix_vStorch', '@welt', '@cdu', '@PaulZiemiak', '@tagesschau', '@YouTube', '@DasErste', '@zeitonline',
            '@HeikoMaas']
for n in droplist:
    df = df[~df['text'].str.contains(n)]
print(len(df), "WITHOUT tweets with tags for popular/political accounts")

# Save main dataset to hard drive
df.to_csv('twitter_data/Twitter_main_final.csv')
