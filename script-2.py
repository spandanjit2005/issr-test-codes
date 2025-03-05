import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('issr-task-1.csv')

# Sentiment Analysis using TextBlob
def get_sentiment(text):
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0.05:
        return 'Positive'
    elif sentiment < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment'] = df['Content'].apply(get_sentiment)

# TF-IDF for high-risk term detection
high_risk_terms = ['suicide', 'kill', 'die', 'end my life', 'no reason to live']
tfidf = TfidfVectorizer(vocabulary=high_risk_terms)
tfidf_matrix = tfidf.fit_transform(df['Content'])

# Risk Level Classification
def get_risk_level(text, tfidf_score):
    if any(term in text.lower() for term in high_risk_terms) or np.sum(tfidf_score) > 0:
        return 'High-Risk'
    elif any(word in text.lower() for word in ['help', 'struggle', 'lost']):
        return 'Moderate Concern'
    else:
        return 'Low Concern'

df['Risk_level'] = df.apply(lambda row: get_risk_level(row['Content'], tfidf_matrix[row.name].toarray()[0]), axis=1)

# Display results
print(df)

# Create a plot showing distribution of posts by sentiment and risk category
plt.figure(figsize=(10, 6))
df.groupby(['Sentiment', 'Risk_level']).size().unstack().plot(kind='bar')
plt.title('Distribution of Posts by Sentiment and Risk Level')
plt.xlabel('Sentiment')
plt.ylabel('Number of Posts')
plt.legend(title='Risk Level')
plt.savefig('script-2.png')

df.to_csv('issr-tast-2.csv')
