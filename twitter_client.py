import tweepy
import pandas as pd
import logging
from datetime import datetime, timedelta
from textblob import TextBlob
import re
import time
from typing import Dict, List, Optional


class TwitterClient:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        try:
            # Twitter API v2 Authentication
            self.client = tweepy.Client(
                bearer_token=config['bearer_token'],
                consumer_key=config['consumer_key'],
                consumer_secret=config['consumer_secret'],
                access_token=config['access_token'],
                access_token_secret=config['access_token_secret'],
                wait_on_rate_limit=True
            )
            self.logger.info("Twitter client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Twitter client: {e}")
            self.client = None

    def get_bitcoin_sentiment(self) -> Dict:
        """Get Bitcoin-related tweets and calculate sentiment"""
        if not self.client:
            return {'sentiment_score': 0, 'tweet_count': 0, 'error': 'Client not initialized'}

        try:
            # Search for Bitcoin-related tweets
            query = '(bitcoin OR BTC OR cryptocurrency) -is:retweet lang:en'
            tweets = tweepy.Paginator(
                self.client.search_recent_tweets,
                query=query,
                max_results=100,
                tweet_fields=['created_at', 'public_metrics']
            ).flatten(limit=500)

            sentiments = []
            tweet_data = []

            for tweet in tweets:
                # Clean tweet text
                clean_text = self._clean_tweet_text(tweet.text)

                # Calculate sentiment
                blob = TextBlob(clean_text)
                sentiment = blob.sentiment.polarity

                sentiments.append(sentiment)
                tweet_data.append({
                    'text': clean_text,
                    'sentiment': sentiment,
                    'created_at': tweet.created_at,
                    'retweet_count': tweet.public_metrics['retweet_count'],
                    'like_count': tweet.public_metrics['like_count']
                })

            if not sentiments:
                return {'sentiment_score': 0, 'tweet_count': 0, 'error': 'No tweets found'}

            # Calculate weighted sentiment (more weight to popular tweets)
            weighted_sentiment = self._calculate_weighted_sentiment(tweet_data)

            return {
                'sentiment_score': weighted_sentiment,
                'tweet_count': len(sentiments),
                'raw_sentiment': sum(sentiments) / len(sentiments),
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error fetching Twitter sentiment: {e}")
            return {'sentiment_score': 0, 'tweet_count': 0, 'error': str(e)}

    def _clean_tweet_text(self, text: str) -> str:
        """Clean tweet text for sentiment analysis"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user mentions and hashtags for cleaner sentiment
        text = re.sub(r'@\w+|#', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def _calculate_weighted_sentiment(self, tweet_data: List[Dict]) -> float:
        """Calculate sentiment weighted by engagement metrics"""
        total_weight = 0
        weighted_sum = 0

        for tweet in tweet_data:
            # Weight based on engagement (likes + retweets)
            weight = 1 + (tweet['like_count'] + tweet['retweet_count'] * 2) / 100
            weighted_sum += tweet['sentiment'] * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0

    def get_trending_topics(self) -> List[str]:
        """Get trending topics related to crypto"""
        if not self.client:
            return []

        try:
            # Get trending topics (requires different API access)
            # This is a simplified version - you might need different endpoints
            trends = []

            # Search for trending crypto hashtags
            crypto_terms = ['#bitcoin', '#btc', '#cryptocurrency', '#crypto']

            for term in crypto_terms:
                try:
                    tweets = self.client.search_recent_tweets(
                        query=f"{term} -is:retweet",
                        max_results=10
                    )
                    if tweets.data:
                        trends.append(term)
                except:
                    continue

            return trends

        except Exception as e:
            self.logger.error(f"Error fetching trending topics: {e}")
            return []