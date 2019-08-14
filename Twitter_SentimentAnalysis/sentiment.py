# Necessary Imports
import nltk
from textblob import TextBlob
import tweepy
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# My Twitter API Authentication Variables
consumer_key =
consumer_secret =
access_token =
access_token_secret =

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
# The hashtag to search for and the no of tweets to take
tweets = api.search('GGMU', count=200)
data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])
# display(data.head(10))
print(tweets[0].created_at)

for tweet in tweets:
    print(tweet.text)


dat = pd.DataFrame(data=[tw.text for tw in tweets], columns=['Tweets'])

for tw in tweets:
    analysis = TextBlob(tw.text)
    print(analysis.sentiment)

nltk.download('punkt')
sid = SentimentIntensityAnalyzer()
listy = []

for index, row in data.iterrows():
    ss = sid.polarity_scores(row["Tweets"])
    listy.append(ss)

se = pd.Series(listy)
data['polarity'] = se.values

display(data.head(100))
