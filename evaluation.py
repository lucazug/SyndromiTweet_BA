
'''''
#  --Bachelorarbeit Luca Zug--
#
#  --evaluation.py
#  Reads RKI and predicted model data, performs data cleaning, calculate the mean positive tweet count for each calendar 
#  week, merges the two dataframes on 'calweek' and prints the resulting dataframe.
#
#  Plots: 
#    * influ_plot : A plot of the number of influenza cases in Germany by calendar week.
#    * influ_log_plot : A plot of the number of influenza cases in Germany by calendar week, with a logarithmic y-axis.
#    * pred_plot : A bar chart of the number of positive case tweets by date, as predicted by the model.
#    * week_counts : A plot of the mean number of positive case tweets by calendar week.
#
#  Please install the necessary packages by running "pip install -r requirements.txt" in the Terminal.
#
'''''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import funcs
import datetime

# Data retrieved from national health institute RKI
rki_data = pd.read_csv("Influenza_RKI.csv", sep=";")

# Data from predictions from model
model_data = pd.read_csv('predicted_Tweets_MAIN.csv', dtype={'Predicted Values':'float',
                                                             'author_id': 'str',
                                                             'username': 'str',
                                                             'author_followers': 'int',
                                                             'author_location':'str',
                                                             'text':'str',
                                                             'created_at': 'str',
                                                             'tweet_location':'str'})
print(model_data.head())

# Preprocessing the manually copied RKI data by dropping unneccessary data points, transposing to plot easily and
# shortening the the calendar weeks from the ftormat 'YYYY-WW' to 'WW'.
influenza_rki = rki_data.iloc[[0]]
influenza_rki = influenza_rki.drop(['Unnamed: 0', "2018-47", "2018-48", "2018-49",  "2018-50",  "2018-51",  "2018-52"],
                                   axis= 1)
temp_influ = influenza_rki
influenza_rki = influenza_rki.transpose()
influenza_rki['calweek'] = temp_influ.columns
influenza_rki['calweek'] = influenza_rki['calweek'].apply(lambda x: int(str(x)[5:]))

influenza_rki = funcs.replaceNaN(influenza_rki, 0)

# Create a semilog plot of the RKI influenza data
influ_log_plot = plt.semilogy(influenza_rki['calweek'], influenza_rki[0])
plt.ylabel('Number of Cases (logarithmic)')
plt.xlabel('Calendar Weeks')
plt.suptitle('Influenza Cases in Germany registered by RKI')
plt.title("(Log Scale)")
plt.xticks(rotation=90)
plt.grid()
plt.show()

# Create a linear plot of the RKI influenza data
influ_plot = plt.plot(influenza_rki['calweek'], influenza_rki[0])
plt.ylabel('Number of Cases')
plt.xlabel('Calendar Weeks')
plt.suptitle('Influenza Cases in Germany registered by RKI')
plt.xticks(rotation=90)
plt.grid()
plt.show()

# Extract the AGI Score data from the RKI data
agi_rki = rki_data.iloc[17, 7:]

# Create a plot of the AGI data
agi_plot = plt.plot(influenza_rki['calweek'], agi_rki)
plt.ylabel('AGI Score')
plt.xlabel('Calendar Weeks')
plt.title('German AGI Score published by RKI')
plt.axhline(115, color='r') # Add a horizontal line at the value of 115, normalised basis value
plt.xticks(rotation=90)
plt.grid()
plt.show()

# Round the predictions in the model data to the nearest integer
model_data['preds'] = model_data['Predicted Values'].round()

# Shorten the date values in the model data by 10 characters
model_data['date'] = funcs.shortenby(10, model_data['created_at'])

# Create a DataFrame with the positive case tweets from the model data
pos_model_data = model_data[model_data['preds'] == 1]
print(f"The number of positive classified tweets is "
      f"{len(pos_model_data)}, {(len(pos_model_data)/len(model_data))*100}% of all tweets.")
print(f"Thus, the number of negative classified tweets is {len(model_data)-len(pos_model_data)}")

# Count the number of positive case tweets by date
counts = pos_model_data.value_counts(['date', 'preds'])
counts = pd.Series.sort_index(counts, axis=0)

# Create a dataframe with unique dates and their corresponding counts
date_counts = pd.DataFrame(np.unique(pos_model_data['date']))
date_counts['counts'] = counts.values
date_counts['seriesdate'] = counts.index
date_counts = date_counts.sort_values(by=0, axis=0)

# Use the cal_week_for_date function to create a new dataframe with the tweet counts by calendar week
week_counts = funcs.cal_week_for_date(date_counts)
weeks = week_counts.groupby('calweek')['counts'].sum()
weeks = weeks.reset_index()
week_counts = pd.DataFrame.drop_duplicates(week_counts, subset= ['calweek'], keep='first')
week_counts = pd.merge(week_counts, weeks, how='inner', left_on='calweek', right_on='calweek', left_index=False,
                       right_index=False)
week_counts = week_counts.drop(['counts_x'], axis=1)

# Create a plot of the positive case tweet counts by date
pred_plot = plt.plot(week_counts['calweek'], week_counts['counts_y'])
plt.ylabel('Number of Positive Case Tweets')
plt.xlabel('Calendar Week')
plt.title('Convolutional Neural Network: Predicted Positive Case Tweets in 2019')
plt.xticks(rotation=90)
plt.show()

# Merge the Model Prediction Counts dataframe with the RKI Influenza dataframe
applied_rki = pd.merge(week_counts, influenza_rki, how='inner', left_on='calweek', right_on='calweek', left_index=False,
                       right_index=False)
applied_rki = applied_rki.drop(['0_x'], axis=1)
applied_rki.rename(columns={'calweek':'calweek', 'counts_y':'pred count', '0_y':'rki count'}, inplace=True)
applied_rki.to_csv(path_or_buf='Weekcounts_Predictions_Eval.csv', index = True, header = True)
print(applied_rki)

# Plot the RKI influenza cases on one axis and the Predicted Tweets on the other y-axis
fig, ax = plt.subplots()
ax.plot(applied_rki['calweek'], applied_rki['rki count'], label='RKI Influenza Cases')
ax2 = ax.twinx()
ax2.plot(applied_rki['calweek'], applied_rki['pred count'], color='red', label='Predicted Cases')
ax.set_xlabel('Calendar Weeks of 2019')
ax.set_ylabel('RKI Influenza Cases')
ax2.set_ylabel('Predicted Cases')
plt.title('RKI Influenza Cases & Predicted Positive Tweets (2019)')
plt.grid()
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.9, 0.87))
plt.show()

# Add the log of the predicted counts to the log_applied_rki dataframe
log_applied_rki = pd.DataFrame()
log_applied_rki['pred count'] = np.log10(applied_rki['pred count'])

# Non-zero value to apply log()
log_applied_rki['rki count'] = applied_rki['rki count'] + 0.001
print(log_applied_rki)

# Add the log of the RKI case counts to the log_applied_rki dataframe
log_applied_rki['rki count'] = np.log10(applied_rki['rki count'])
log_applied_rki['calweek'] = applied_rki['calweek']

# Plot the log values for the RKI Influenza Cases and the Predicted Tweets Cases
fig, ax = plt.subplots()
ax.plot(log_applied_rki['calweek'], log_applied_rki['rki count'], label='RKI Influenza Cases')
ax2 = ax.twinx()
ax2.plot(log_applied_rki['calweek'], log_applied_rki['pred count'], color='red', label='Predicted Cases')
ax.set_xlabel('Calendar Weeks of 2019')
ax.set_ylabel('RKI Influenza Cases (log)')
ax2.set_ylabel('Predicted Cases (log)')
plt.suptitle('RKI Influenza Cases & Predicted Positive Tweets (2019)')
plt.title('(Log Scale)')
plt.grid()
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.9, 0.87))
plt.show()

# Plot the log values for the RKI Influenza Cases and the Predicted Tweets Cases
fig, ax = plt.subplots()
ax.plot(log_applied_rki['calweek'], agi_rki, label='RKI AGI Score')
ax.axhline(115, color='r')
ax2 = ax.twinx()
ax2.plot(log_applied_rki['calweek'], log_applied_rki['pred count'], color='red', label='Predicted Cases')
ax.set_xlabel('Calendar Weeks of 2019')
ax.set_ylabel('AGI Score')
ax2.set_ylabel('Predicted Cases (log)')
plt.suptitle('RKI AGI Score & Predicted Positive Tweets (2019)')
plt.title('(Log Scale for Predicted Cases)')
plt.grid()
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.9, 0.87))
plt.show()

# Normalise both columns in the data (predictions and RKI data) to make more comparable
norm_applied_rki = funcs.normalise(applied_rki, "pred count", "rki count")
print(norm_applied_rki)

## Correlation analysis
# Print the correlation between the predicted counts and the RKI counts
print('Correlation between Predictions for Weeks and RKI Data')
print(applied_rki['pred count'].corr(applied_rki['rki count']).round(3))

# TRIED: Cross-correlation for time series. No useful information returned.
corr = np.correlate(applied_rki['pred count'], applied_rki['rki count'], mode='full')
print(corr.astype(int))

# Print the correlation between the AGI Score and the RKI counts
applied_rki['agi'] = agi_rki.values
applied_rki['agi'] = applied_rki['agi'].astype(int)

print('Correlation between Predictions for Weeks and AGI Practice Score Data')
print(applied_rki['pred count'].corr(applied_rki['agi']).round(3))
