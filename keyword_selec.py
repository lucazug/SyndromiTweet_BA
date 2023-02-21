'''''
#  --Bachelorarbeit Luca Zug--
#
#  --keyword_selec.py
#  Read list of symptoms, create Twitter API search query and call tweets through the API, convert result to DataFrame
#  and output 1/8 of the result randomly for manual labelling for the keyword selection process.
#
'''''

import pandas as pd
import tweepy as twp
import time
import json
import funcs
import numpy as np
import random

symp = pd.read_csv('Symptome.csv')
symp = symp.drop_duplicates()

# Format the list of symptoms into a list of strings, which are joined with ' -is:retweet OR ' to form a single string,
# which forms the search query.
x = symp.to_string(header=False, index=False, index_names=False).split('\n')
x = [' '.join(ele.split()) for ele in x]
symptomlist = [' -is:retweet OR '.join(x)]
tweet_query = symptomlist[0] + ' -has:media'
print(tweet_query)

# Initialise a Twitter API client using a bearer token
#bearer_token = ## HIDDEN ##
client = twp.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

# Generate a list of 50 dates with random times of day as start and end times for the API calls.
time_interval = funcs.random_time_interval(50, 2019, 1, 1, 30, 5)

# Iterates over the list of dictionaries 'time_interval' and uses the API client to search for tweets using 'tweet_query'
# and the start and end times specified in the dictionary. The resulting tweets are stored in the list 'ili_tweets'.
ili_tweets = []
for t in range(0,len(time_interval)):
    start_time = time_interval['start_date'][t]
    end_time = time_interval['end_date'][t]

    print(start_time, end_time)
    try:
        for response in twp.Paginator(client.search_all_tweets,
                                      query=tweet_query,
                                      user_fields=['username', 'public_metrics', 'description', 'location'],
                                      tweet_fields=['created_at', 'geo', 'text'],
                                      expansions='author_id',
                                      start_time=start_time,
                                      end_time=end_time,
                                      max_results=500):
            time.sleep(1)
            ili_tweets.append(response)
            print(response)
    except twp.errors.TwitterServerError as e:
        print("Tweepy Error: {}".format(e))
        pass

result = []
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
        result.append({'author_id': tweet.author_id,
                       'username': author_info['username'],
                       'author_followers': author_info['followers'],
                       'author_tweets': author_info['tweets'],
                       'author_description': author_info['description'],
                       'author_location': author_info['location'],
                       'text': tweet.text,
                       'created_at': tweet.created_at,
                       'tweet_location': tweet.geo
                       })

# Converts tweet data list into a Pandas DataFrame and saves it to a file called 'Twitter_keyword.csv'
df = pd.DataFrame(result)
df.to_csv('Twitter_keyword.csv')

# Randomly selects 1/8 of the rows in the DataFrame and saves them to a file called 'Twitter_Keyword_Coded.csv' for
# manual labelling for the keyword selection process.
num_results = df.shape[0]
print(df.shape[0])
eigth_result = num_results / 8
selectlist = []
selectlist = random.sample(range(num_results), round(eigth_result))

df_manual = df.loc[selectlist]
df_manual.to_csv('twitter_data/Twitter_Keyword_Coded.csv')
