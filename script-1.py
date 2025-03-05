import praw
import csv
import re
import nltk
from datetime import datetime
from nltk.corpus import stopwords

nltk.download('stopwords')

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id=${{secrets.CLIENT_ID}},
    client_secret=${{secrets.CLIENT_SECRET}},
    user_agent='u/equesinferi_'
)

# Define keywords for filtering
keywords = [
    'depressed', 'depression', 'anxiety', 'addiction', 'suicidal',
    'overwhelmed', 'substance abuse', 'mental health', 'crisis',
    'panic attack', 'self-harm', 'bipolar', 'ptsd', 'ocd', 'trauma'
]

# Define subreddits to search
subreddits = ['mentalhealth', 'depression', 'anxiety', 'SuicideWatch', 'addiction']

# Function to preprocess text
def preprocess_text(text):

    text = text.lower()
    
    words = text.split()
    words = [word for word in words if not word.startswith('http') and not word.startswith('www.')]
    
    cleaned_words = []
    for word in words:
        cleaned_word = ''.join(char for char in word if char.isalpha())
        if cleaned_word:
            cleaned_words.append(cleaned_word)
    
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in cleaned_words if word not in stop_words]
    
    return ' '.join(filtered_words)

# Open CSV file for writing
with open('issr-task-1.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Post ID', 'Timestamp', 'Content', 'Likes', 'Comments', 'Subreddit']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for subreddit_name in subreddits:
        subreddit = reddit.subreddit(subreddit_name)
        
        for keyword in keywords:
            for post in subreddit.search(keyword, limit=10):
                content = preprocess_text(post.title + ' ' + post.selftext)
                
                if keyword in content:
                    writer.writerow({
                        'Post ID': post.id,
                        'Timestamp': datetime.fromtimestamp(post.created_utc),
                        'Content': content,
                        'Likes': post.score,
                        'Comments': post.num_comments,
                        'Subreddit': subreddit_name
                    })

print("Data extraction complete.")
