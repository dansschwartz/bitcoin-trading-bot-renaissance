import praw
import pandas as pd
import logging
from datetime import datetime, timedelta
from textblob import TextBlob
import re
from typing import Dict, List, Optional


class RedditClient:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        try:
            self.reddit = praw.Reddit(
                client_id=config['client_id'],
                client_secret=config['client_secret'],
                user_agent=config['user_agent'],
                username=config.get('username'),
                password=config.get('password')
            )

            # Test connection
            self.reddit.user.me()
            self.logger.info("Reddit client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Reddit client: {e}")
            self.reddit = None

    def get_crypto_sentiment(self) -> Dict:
        """Get sentiment from crypto-related subreddits"""
        if not self.reddit:
            return {'sentiment_score': 0, 'post_count': 0, 'error': 'Client not initialized'}

        try:
            subreddits = ['Bitcoin', 'CryptoCurrency', 'BitcoinMarkets', 'btc']
            all_posts = []

            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)

                    # Get hot posts
                    for post in subreddit.hot(limit=25):
                        if self._is_bitcoin_related(post.title + " " + post.selftext):
                            all_posts.append({
                                'title': post.title,
                                'text': post.selftext,
                                'score': post.score,
                                'upvote_ratio': post.upvote_ratio,
                                'num_comments': post.num_comments,
                                'created_utc': post.created_utc,
                                'subreddit': subreddit_name
                            })

                except Exception as e:
                    self.logger.warning(f"Error accessing subreddit {subreddit_name}: {e}")
                    continue

            if not all_posts:
                return {'sentiment_score': 0, 'post_count': 0, 'error': 'No posts found'}

            # Calculate sentiment for each post
            sentiments = []
            for post in all_posts:
                text = f"{post['title']} {post['text']}"
                clean_text = self._clean_text(text)

                if len(clean_text) > 10:  # Only analyze substantial text
                    blob = TextBlob(clean_text)
                    sentiment = blob.sentiment.polarity

                    # Weight by post popularity
                    weight = 1 + (post['score'] * post['upvote_ratio']) / 100
                    sentiments.append({
                        'sentiment': sentiment,
                        'weight': weight,
                        'score': post['score'],
                        'comments': post['num_comments']
                    })

            if not sentiments:
                return {'sentiment_score': 0, 'post_count': 0, 'error': 'No valid sentiment data'}

            # Calculate weighted sentiment
            weighted_sentiment = sum(s['sentiment'] * s['weight'] for s in sentiments) / sum(
                s['weight'] for s in sentiments)

            return {
                'sentiment_score': weighted_sentiment,
                'post_count': len(sentiments),
                'average_score': sum(s['score'] for s in sentiments) / len(sentiments),
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error fetching Reddit sentiment: {e}")
            return {'sentiment_score': 0, 'post_count': 0, 'error': str(e)}

    def _is_bitcoin_related(self, text: str) -> bool:
        """Check if text is Bitcoin-related"""
        bitcoin_keywords = ['bitcoin', 'btc', 'cryptocurrency', 'crypto', 'satoshi']
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in bitcoin_keywords)

    def _clean_text(self, text: str) -> str:
        """Clean text for sentiment analysis"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def get_discussion_volume(self) -> Dict:
        """Get discussion volume metrics"""
        if not self.reddit:
            return {'volume_score': 0, 'error': 'Client not initialized'}

        try:
            subreddits = ['Bitcoin', 'CryptoCurrency', 'BitcoinMarkets']
            total_posts = 0
            total_comments = 0

            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)

                    for post in subreddit.hot(limit=50):
                        if self._is_bitcoin_related(post.title):
                            total_posts += 1
                            total_comments += post.num_comments

                except Exception as e:
                    continue

            # Calculate volume score (posts + weighted comments)
            volume_score = total_posts + (total_comments * 0.1)

            return {
                'volume_score': volume_score,
                'total_posts': total_posts,
                'total_comments': total_comments,
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error fetching discussion volume: {e}")
            return {'volume_score': 0, 'error': str(e)}