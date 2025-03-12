import os
from dotenv import load_dotenv
import praw
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Load environment variables from .env file
load_dotenv()

# Download VADER Lexicon
nltk.download('vader_lexicon')

# Reddit API Credentials
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_SECRET = os.getenv('REDDIT_SECRET')
USER_AGENT = os.getenv('USER_AGENT')

# Initialize Reddit API
reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID,
                     client_secret=REDDIT_SECRET,
                     user_agent=USER_AGENT)

# Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

def get_posts_and_comments(subreddit_name, post_limit=10, comment_limit=10):
    subreddit = reddit.subreddit(subreddit_name)
    posts_data = []

    for submission in subreddit.new(limit=post_limit):
        submission.comments.replace_more(limit=0)
        comments = [comment.body for comment in submission.comments[:comment_limit]]
        posts_data.append({
            'title': submission.title,
            'comments': comments
        })

    return posts_data

def analyze_sentiment(text):
    sentiment_score = sia.polarity_scores(text)["compound"]
    if sentiment_score > 0:
        sentiment = "Positive"
    elif sentiment_score < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, sentiment_score

def main():
    subreddit_name = input("Enter subreddit name: ")
    posts_data = get_posts_and_comments(subreddit_name)

    for idx, post in enumerate(posts_data, start=1):
        post_sentiment, post_score = analyze_sentiment(post['title'])
        print(f"\nPost {idx}:")
        print(f"Title: {post['title']}")
        print(f"Post Sentiment: {post_sentiment} (Score: {post_score})")

        for c_idx, comment in enumerate(post['comments'], start=1):
            comment_sentiment, comment_score = analyze_sentiment(comment)
            print(f"\n\tComment {c_idx}: {comment}")
            print(f"\tComment Sentiment: {comment_sentiment} (Score: {comment_score})")

if __name__ == "__main__":
    main()
