#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tweepy
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[2]:


#My Twitter API Authentication Variables
consumer_key = '71dsR93Fji2k5zJs5VLrPJiJI'
consumer_secret = 'orSafrcav6FiHuzNq97vOujbW3IYqfV51xjugV1q920PFFxTOz'
access_token = '1135525772565929984-KctR1ei9KbVWve7H27q6HypRYFsBRO'
access_token_secret = 'dijM32Xx7RBX5GfsT0Tsgr11gIrtylVYNWCHCVHNI4Ist'


# In[3]:


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

tweets = api.search('Liverpool', count=200)


data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])

#display(data.head(10))


print(tweets[0].created_at)


# In[4]:


for tweet in tweets:
    print(tweet.text)


# In[5]:


from textblob import TextBlob


# In[6]:


dat = pd.DataFrame(data=[tw.text for tw in tweets], columns=['Tweets'])


# In[7]:


dat


# In[9]:


for tw in tweets:
    analysis= TextBlob(tw.text)
    print(analysis.sentiment)
    


# In[10]:


analysis.sentiment.polarity


# In[11]:


import nltk
nltk.download('punkt')


# In[12]:


sid = SentimentIntensityAnalyzer()


listy = []

for index, row in data.iterrows():
  ss = sid.polarity_scores(row["Tweets"])
  listy.append(ss)
  
se = pd.Series(listy)
data['polarity'] = se.values

display(data.head(100))

