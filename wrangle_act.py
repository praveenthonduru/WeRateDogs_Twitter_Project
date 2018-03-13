
# coding: utf-8

# # #WeRateDogs - Data Wrangling & Analyzing Twitter Data Project

# 
# 
# By Praveen Thonduru
# 
# Date: March 5th, 2018
# # Introduction
# 
# The goal of this project is to wrangle the WeRateDogs Twitter data to create interesting and trustworthy analyses and visualizations. The challenge lies in the fact that the Twitter archive is great, but it only contains very basic tweet information that comes in JSON format. For a successful project, I needed to gather, asses and clean the Twitter data for a worthy analysis and visualization.
# 

# In[2]:


#import major libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Gathering Data

# In[3]:


# Read the twitter_archive_enhanced.csv file and 
# set it as a dataframe called df.
df = pd.read_csv("twitter-archive-enhanced.csv")
df.head(2)


# In[4]:


# Programmatically download the dog image prediction files from 
# the Udacity server using Request library
import os
import requests

# Save to a file
folder_name = 'image_predictions'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

url = 'https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv'
response = requests.get(url)
response

with open(os.path.join(folder_name,
                      url.split('/')[-1]), mode='wb') as file:
    file.write(response.content)


# In[5]:


#open tsv file
images = pd.read_table('image_predictions/image-predictions.tsv',
                       sep='\t')


# Query the Twitter API for each tweet's JSON data using Python's Tweepy library and store each tweet's entire set of JSON data in a file.

# In[6]:


#Importing libraries
import tweepy
from tweepy import OAuthHandler
import json
import csv
import sys
import os
import time


# authentication pieces
consumer_key = ''
consumer_secret = ''
access_token = ''
access_secret = ''


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth_handler=auth, 
                 wait_on_rate_limit=True, 
                 wait_on_rate_limit_notify=True)


# In[7]:


#Twitter Query using tweet_id information from the df.

tweet_ids = list(df.tweet_id)

tweet_data = {}
for tweet in tweet_ids:
    try:
        tweet_status = api.get_status(tweet,
                                      wait_on_rate_limit=True, 
                                      wait_on_rate_limit_notify=True)
        tweet_data[str(tweet)] = tweet_status._json
    except: 
        print("Error for: " + str(tweet))


# In[8]:


import json

with open('tweet_json.txt', 'w') as outfile:  
    json.dump(tweet_data, outfile, 
              sort_keys = True,
              indent=4,
              ensure_ascii = False)


# In[9]:


tweet_df = pd.read_json('tweet_json.txt',orient='index')


# 
# # Assessing data
# 
# After gathering each of the above pieces of data, assess them visually and programmatically for quality and tidiness issues. Detect and document at least eight (8) quality issues and two (2) tidiness issues in your wrangle_act.ipynb Jupyter Notebook. To meet specifications, the issues that satisfy the Project Motivation (see the Key Points header on the previous page) must be assessed.
# 

# In[10]:


df.info()


# In[11]:


images.info()


# In[12]:


tweet_df.info()


# ### Quality Issues
# 
# Issues with the data's content
# 
# Need to remove rows that have been retweeted, therby they are not original tweets
# 
# ### A tweet that has been retweeted
# 
# The relevant field is retweet_count. This field provides the number of times this tweet was retweeted. Note that this number may vary over time, as additional people retweet the tweet.
# 
# ### A tweet that is a retweet
# 
# Want to exclude any tweet that is a retweet. Two fields are significant. First, the retweeted_status contains the source tweet (i.e., the tweet that was retweeted). The present or absence of this field can be used to identify tweets that are retweets. Second, the retweet_count is the count of the retweets of the source tweet, not this tweet.
# 
# Therefore I will isolate all rows in the retweeted_status column that have a value and delete it from the dataframe. This will remove tweets that are a retweet from the dataframe.
# 
# #### Dataframe Table:
# 
#   - Names of dogs are miss labelled, mispelled or missing. Cross-reference text data with Names column.
#   - Excluded columns from dataset that are not needed for the analysis
#   - 181 records have a retweeted_status_id, these will need to be exluded from the dataset
# 
# ### Image Predictions Table:
# 
#    - p1 column: capitalize the first letter of each word, make consistent
#    - p2 column: capitalize the first letter of each word, make consistent
#    - p3 column: capitalize the first letter of each word, make consistent
#    - Remove the (_) between the words
# 
# #### Tweet_DF Table:
# 
#    - rename the id column to "tweet_id" to match the other 2 tables
#    - 176 records have a retweeted_status, will need to be excluded
# 
# ## Tidyness Issues
# 
# Issues with the structure of the data
# 
# #### Dataframe Table:
# 
#    - Parse the datetime information into seperate columns
#    - Drop columns that are not needed & rearrange column order for an easier read
#    - Combine each dog stage column into a single column named "stage"
#    - tweet_id column needs to be converted from a number to string value
#    - Date and Time columns need to be converted to datetime objects
#    - Rating columns need to be converted to float values
# 
# #### Tweet_DF Table:
# 
#    - convert id column from a number to a string
#    - Reindex the tweet_df table using the tweet_id
#    - Change column order in the Tweet_df and the df tables for an easier read of the data
#    - Consolidate the tweet_df table into the following columns: tweet_id, retweet count, favorite count,text
# 
# #### All Tables:
# 
#    - perform an inner join between all three datasets
# 
# 

# # Cleaning data

# In[13]:


# Copy the dataframes 
df_clean = df.copy()
images_clean = images.copy()
tweet_df_clean = tweet_df.copy()


# ### DF_CLEAN dataframe
# ##### Define
# 
# Missing Data: replace faulty names or corrected names
# 
# #### Code
# 

# In[14]:


# Missing Data
# replacing faulty names with None value or corrected Name
df_clean['name'].replace('the', 'None', inplace=True)
df_clean['name'].replace("light",'None', inplace=True)
df_clean['name'].replace("life",'None', inplace=True)
df_clean['name'].replace("an",'None', inplace=True)
df_clean['name'].replace("a",'None', inplace=True)
df_clean['name'].replace("by",'None', inplace=True)
df_clean['name'].replace("actually",'None', inplace=True)
df_clean['name'].replace("just",'None', inplace=True)
df_clean['name'].replace("getting",'None', inplace=True) 
df_clean['name'].replace("infuriating",'None', inplace=True) 
df_clean['name'].replace("old",'None', inplace=True) 
df_clean['name'].replace("all",'None', inplace=True) 
df_clean['name'].replace("this",'None', inplace=True) 
df_clean['name'].replace("very",'None', inplace=True) 
df_clean['name'].replace("mad",'None', inplace=True) 
df_clean['name'].replace("not",'None', inplace=True)
df_clean['name'].replace("one",'None', inplace=True)
df_clean['name'].replace("my",'None', inplace=True)
df_clean['name'].replace("O","O'Malley", inplace=True)
df_clean['name'].replace("quite","None", inplace=True)
df_clean['name'].replace("such","None", inplace=True)


# ### Test

# In[15]:


df_clean


# 
# #### Define
# 
# Identify and exlude tweets that have a retweeted_status because the tweet is a retweet and therefore not original.
# 
# ##### Code

# In[16]:


# Identify how many tweets are retweets by the "retweeted_status" columns
df_clean.info()


# In[17]:


df_clean[df_clean['retweeted_status_id'].notnull()==True]


# In[18]:


# remove these values from the dataframe using the drop() function
df_clean.drop(df_clean[df_clean['retweeted_status_id'].notnull()== True].index,inplace=True)


# ### Test

# In[19]:


df_clean.info()


# 
# #### Define
# 
# Convert the timestamp column from a string to DateTime objects
# 
# #### Code

# In[20]:


# DF table: Clean timestamp column
from datetime import datetime,timedelta

#what data type is the timestamp currently in?
type(df_clean['timestamp'].iloc[0])


# In[21]:


#Use pd.to_datetime to convert the column from strings to DateTime objects.
df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])

#Now that the timestamp column are actually DateTime objects, use .apply() to create 3 new columns 
#called Hour, Day, Month, and Year. You will create these columns based off of the timeStamp column, 
#reference the solutions if you get stuck on this step.

df_clean['date'] = df_clean['timestamp'].apply(lambda time: time.strftime('%m-%d-%Y'))
df_clean['time'] = df_clean['timestamp'].apply(lambda time: time.strftime('%H:%M'))


# #### Test

# In[22]:


df_clean.head(1)


# 
# #### Define
# 
# Combine the Dog stages into one column names "Stages"
# 
# ##### Code

# In[23]:


df_clean['stage'] = df[['doggo', 'floofer','pupper','puppo']].apply(lambda x: ''.join(x), axis=1)

df_clean['stage'].replace("NoneNoneNoneNone","None ", inplace=True)
df_clean['stage'].replace("doggoNoneNoneNone","doggo", inplace=True)
df_clean['stage'].replace("NoneflooferNoneNone","floofer", inplace=True)
df_clean['stage'].replace("NoneNonepupperNone","pupper", inplace=True)
df_clean['stage'].replace("NoneNoneNonepuppo","puppo", inplace=True)


# #### Test

# In[24]:


df_clean


# 
# #### Define
# 
# Remove unwanted columns from the df_clean dataset and rearange columns for an easier read
# 
# #### Code

# In[25]:


# remove unwanted columns from df_clean columns
df_clean.drop(['timestamp',
               'retweeted_status_user_id',
               'retweeted_status_id',
               'retweeted_status_timestamp',
               'in_reply_to_status_id',
               'in_reply_to_user_id',
               'in_reply_to_status_id',
               'expanded_urls',
               'source',
               'doggo',
               'floofer',
               'pupper',
               'puppo',
               'text'], axis=1,inplace=True)


# In[26]:


df_clean.head()


# In[27]:


# Change the order (the index) of the df_clean columns
columnTitles = ['tweet_id', 
                'date', 
                'time',
                'name',
                'stage',
                'rating_numerator',
                'rating_denominator']
df_clean = df_clean.reindex(columns=columnTitles)


# #### Test

# In[28]:


df_clean.head()


# 
# #### Define
# 
# tweet_id column should be objects (i.e. how pandas represents strings), not integers or floats because it is not numeric. Update this field to strings
# #### Code

# In[29]:


df_clean.info()


# In[30]:


df_clean.tweet_id = df_clean.tweet_id.astype(str)


# #### Test

# In[32]:


type(df_clean['tweet_id'].iloc[0])


# 
# #### Define
# 
# Date and Time columns should be datetime objects to easily perform time-related calculations.
# #### Code

# In[33]:


df_clean['date'] = pd.to_datetime(df_clean['date'])
df_clean['time'] = pd.to_datetime(df_clean['time'])


# #### Test

# In[34]:


type(df_clean['date'].iloc[0])
type(df_clean['time'].iloc[0])


# 
# #### Define
# 
# rating_numerator and rating_denominator field needs to be converted to float as there is nothing stopping future dog ratings from having a number with a decimal.
# #### Code
# 

# In[35]:


df_clean['rating_numerator'] = df_clean['rating_numerator'].astype(float)
df_clean['rating_denominator'] = df_clean['rating_denominator'].astype(float)


# #### Test

# In[36]:


type(df_clean['rating_numerator'].iloc[0])


# In[37]:


df_clean.info()


# 
# ## TWEET_DF_CLEAN dataframe
# #### Define
# 
# Rename the "id" column to "tweet_id" to match the other 2 datasets
# #### Code

# In[62]:


tweet_df_clean = tweet_df.copy()


# In[63]:


tweet_df_clean.columns


# In[64]:


tweet_df_clean.rename(columns={'id': 'tweet_id'}, inplace=True)


# #### Test

# In[65]:


tweet_df_clean.columns


# In[66]:


tweet_df_clean.head(1)


# 
# #### Define
# 
# convert tweet_id column from a number to a string value
# #### Code

# In[67]:


tweet_df_clean.tweet_id = tweet_df_clean.tweet_id.astype(str)


# #### Test

# In[68]:


type(df_clean['tweet_id'].iloc[0])


# 
# #### Define
# 
# Identify and exlude tweets that have a retweeted_status because the tweet is a retweet and therefore not original.
# #### Code

# In[69]:


# Identify how many tweets are retweets by the "retweeted_status" columns
tweet_df_clean.info()


# In[70]:


#Single out the non-null values in the 'retweet_status' column
tweet_df_clean[tweet_df_clean['retweeted_status'].notnull()==True]


# Notice, if a tweet was retweeted, there with be a "RT @" in the text.

# In[71]:


#remove the tweets that are retweets from the dataset
tweet_df_clean.drop(tweet_df_clean[tweet_df_clean['retweeted_status'].notnull()== True].index,inplace=True)


# #### Test

# In[72]:


tweet_df_clean.info()


# In[73]:


tweet_df_clean.head(1)


# 
# #### Define
# 
# remove unwanted columns from the dataframe
# #### Code

# In[74]:


tweet_df_clean.drop(['contributors',
                     'coordinates',
                     'created_at',
                     'entities',
                     'extended_entities',
                     'favorited',
                     'geo',
                     'id_str',
                     'in_reply_to_screen_name',
                     'in_reply_to_status_id',
                     'in_reply_to_status_id_str',
                     'in_reply_to_user_id',
                     'in_reply_to_user_id_str',
                     'is_quote_status',
                     'lang',
                     'place',
                     'possibly_sensitive',
                     'possibly_sensitive_appealable',
                     'quoted_status',
                     'quoted_status_id',
                     'quoted_status_id_str',
                     'retweeted',
                     'retweeted_status',
                     'source',
                     'truncated',
                     'user'],axis=1,inplace=True)


# #### Test

# In[75]:


tweet_df_clean.head(10)


# 
# ### Image Predictions dataframe
# #### Define
# 
# Replace the underscore in the p1,p2,p3 columns
# #### Code

# In[76]:


images_clean['p1'] = images_clean['p1'].str.replace('_', ' ')
images_clean['p2'] = images_clean['p2'].str.replace('_', ' ')
images_clean['p3'] = images_clean['p3'].str.replace('_', ' ')


# #### Test

# In[77]:


images_clean.head()


# 
# #### Define
# 
# Make the text consistent and pretty
# #### Code
# 

# In[78]:


images_clean['p1'] = images_clean['p1'].str.title()
images_clean['p2'] = images_clean['p2'].str.title()
images_clean['p3'] = images_clean['p3'].str.title()


# #### Test

# In[79]:


images_clean.head()


# 
# #### Define
# 
# Convert "tweet_id" column to string value
# #### Code

# In[80]:


images_clean.tweet_id = images_clean.tweet_id.astype(str)


# #### Test

# In[81]:


type(images_clean['tweet_id'].iloc[0])


# 
# #### Define
# 
# Use an inner join to combine the datasets together into a single dataframe
# #### Code

# In[82]:


df_merge = pd.merge(df_clean, tweet_df_clean,on='tweet_id', how='inner')


# In[83]:


df_merge = pd.merge(df_merge, images_clean,on='tweet_id', how='inner')


# #### Test

# In[85]:


df_merge


# 
# #### Define
# 
# Fix the "Date & Time" columns in the new df_merge dataframe.
# 
#    - rearrange the Date: month-day-year
#    - drop the date info from the Time column
# 
# #### Code

# In[86]:


df_merge['date'] = df_merge['date'].apply(lambda time: time.strftime('%m-%d-%Y'))
df_merge['time'] = df_merge['time'].apply(lambda time: time.strftime('%H:%M'))


# #### Test

# In[87]:


df_merge


# 
# ### Storing, Analyzing, and Visualizing Data for this Project
# 
# Store the clean DataFrame(s) in a CSV file with the main one named twitter_archive_master.csv. If additional files exist because multiple tables are required for tidiness, name these files appropriately. Additionally, you may store the cleaned data in a SQLite database (which is to be submitted as well if you do).
# 
# Analyze and visualize your wrangled data in your wrangle_act.ipynb Jupyter Notebook. At least three (3) insights and one (1) visualization must be produced.
# #### Store the Cleaned dataframes

# In[88]:


# Save to a file
folder_name = 'Final_Documents'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    
df_clean.to_csv('Final_Documents/twitter_archive_master.csv')
images_clean.to_csv('Final_Documents/image_prediction_master.csv')
tweet_df_clean.to_csv('Final_Documents/tweet_query_master.csv')
df_merge.to_csv('Final_Documents/final_master.csv')


# ### Analyze

# In[89]:


# Read in the updated dataframes
df_master= pd.read_csv("Final_Documents/final_master.csv")
image_pred_df = pd.read_csv("Final_Documents/image_prediction_master.csv")


# ##### Let's take a look at the cleaned dataframe

# In[90]:


df_master.head()


# In[91]:


df_master.info()


# 
# #### Define
# 
#    - drop the "unnamed : 0" column
#    - convert the "tweet_id" column to str
# 
# #### Code
# 

# In[92]:


df_master.drop(['Unnamed: 0'],axis=1,inplace=True)
df_master.tweet_id = df_master.tweet_id.astype(str)


# #### Test

# In[93]:


df_master.info()


# ### Descriptive Statistical Analysis

# In[94]:


# Descriptive statistics
print('\n')
print("Descriptive statistics of the dataset:")
stats= df_master.drop(['tweet_id'], axis=1)

stats.describe()


# 
# 
# ### Key Takeaways:
# 
#    - The neural network performed the best on the 1st iteration with a mean prediciton of 0.587
#    - Mean rating for a dog was 12.843/10 with an outlier of 1776/10
#    - Mean retweet count for an original tweet was 2576 and a maximum value of 61935.
#    - Mean favorite count for an original tweet was 8373 and a maximum value of 123165.
# 
# ##### Questions:
# 
# Is the most popular tweet, which has a maximum favorite count of 123067, the same tweet with the highest retweet count of 61900?
# 
# Investigate the rating_numerator outlier further. What can we learn from it?
# #### Define
# 
# Explore the rating_numerator outlier
# #### Code
# 

# In[95]:


df_master[df_master['rating_numerator']==1776]


# In[96]:


#Let's pull his picture the dataset
df_master[df_master['tweet_id']==749981277374128128].jpg_url


# 
# #### Results
# 
# Turns out the outlier in the rating_numerator data is an awesome dog named Atticus who loves celebrating America's birthday!
# Define
# 
#    - Who has the most favorited dog?
#    - What does their picture look like?
#    - Does this dog also have the most retweets?
# 
# 

# In[98]:


df_master[df_master["favorite_count"]== 123705]


# In[99]:


#Let's pull his picture the dataset
df_master[df_master['tweet_id']==807106840509214720].jpg_url


# 
# #### Results
# 
# The same dog has both the highest favorite and retweet count! His name is Stephan and I think the neural network predicted it right, he looks to be a Chihuahua mix!
# #### Define
# 
#    - What are the top 5 most popular dog names?
# 
# #### Code
# 

# In[100]:


# What are the top 5 most common dog names?
from collections import Counter

x = df_master['name']

count = Counter(x)
count.most_common(5)


# In[101]:


count.most_common(6)


# 
# #### Results
# 
# Top dog names are Oliver, Winston, Tucker, Penny and Cooper
# #### Define
# 
# What is the most common dog rating?
# #### Code

# In[102]:


x = df_master['rating_numerator']
count = Counter(x)
count.most_common()


# 
# #### Results
# 
# Most common Dog Rating is 10 with 304 instances
# #### Define
# 
# Explore the dogs with the lowest dog rating
# #### Code

# In[103]:


#Lets take a look at the lowest rated dogs 
df_master[df_master['rating_numerator']==1]


# In[105]:


#Let's pull the picture the dataset
df_master[df_master['tweet_id']==675153376133427200].jpg_url  #Poodle puppo that blends into the carpet


# In[106]:


#Let's pull the picture the dataset
df_master[df_master['tweet_id']==667549055577362432].jpg_url # picture of a fan


# In[107]:


#Let's pull the picture the dataset
df_master[df_master['tweet_id']==666287406224695296].jpg_url # poor doggo with a hurt leg


# In[108]:


#Let's pull the picture the dataset
df_master[df_master['tweet_id']==666104133288665088].jpg_url # picture of a chicken


# ### Visualization

# In[112]:


#Visualization comparing the favorite & retweet counts
sns.set_style('whitegrid')
g = sns.jointplot(x="retweet_count", 
                  y="favorite_count", 
                  data=df_master, 
                  color="blue",
                  kind="kde", 
                  size=10)
g.plot_joint(plt.scatter,  
             c="black",
             s=80, 
             linewidth=1, 
             marker="+",
             alpha=0.45)
g.set_axis_labels("Retweet Count", "Favorite Count",fontsize=12)

g.fig.subplots_adjust(top=0.9)
g.suptitle('Is there a correlation between the retweet & favorite counts?',
                 fontsize=16)

