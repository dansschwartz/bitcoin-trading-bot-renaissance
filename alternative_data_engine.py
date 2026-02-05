"""
Alternative Data Engine - Live Fear & Greed + Social Sentiment
Provides a lightweight, production-friendly alternative data signal stream.
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, Any, List

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False
    tweepy = None

try:
    import praw
    REDDIT_AVAILABLE = True
except ImportError:
    REDDIT_AVAILABLE = False
    praw = None

try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False
    NewsApiClient = None

try:
    from textblob import TextBlob
    SENTIMENT_BACKEND = "textblob"
except ImportError:
    TextBlob = None
    SENTIMENT_BACKEND = "vader"

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    SentimentIntensityAnalyzer = None
    VADER_AVAILABLE = False


@dataclass
class AlternativeSignal:
    """Container for alternative data signals"""
    social_sentiment: float = 0.0  # -1 to 1
    reddit_sentiment: float = 0.0  # -1 to 1
    news_sentiment: float = 0.0  # -1 to 1
    on_chain_strength: float = 0.0  # -1 to 1 (0 if unavailable)
    market_psychology: float = 0.0  # -1 to 1
    confidence: float = 0.0  # 0 to 1
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FearGreedClient:
    """Fetches the Fear & Greed index."""

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.api_url = config.get("api_url", "https://api.alternative.me/fng/")

    def fetch(self) -> Optional[Dict[str, Any]]:
        if not REQUESTS_AVAILABLE:
            self.logger.warning("requests not available; skipping Fear & Greed fetch")
            return None
        try:
            response = requests.get(self.api_url, timeout=10)
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data", []) if isinstance(payload, dict) else []
            if not data:
                return None
            value = int(data[0].get("value", 50))
            return {
                "value": value,
                "timestamp": data[0].get("timestamp")
            }
        except Exception as exc:
            self.logger.warning("Fear & Greed fetch failed: %s", exc)
            return None


class TwitterSentimentClient:
    """Fetches recent Twitter sentiment using the v2 API."""

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        bearer_env = config.get("twitter_bearer_env", "TWITTER_BEARER_TOKEN")
        self.bearer_token = config.get("bearer_token") or os.getenv(bearer_env, "").strip()
        self.query = config.get("twitter_query", "(bitcoin OR BTC) -is:retweet lang:en")
        self.max_tweets = int(config.get("max_tweets", 100))

        self.client = None
        if self.bearer_token and TWEEPY_AVAILABLE:
            try:
                self.client = tweepy.Client(bearer_token=self.bearer_token, wait_on_rate_limit=True)
            except Exception as exc:
                self.logger.warning("Failed to initialize Tweepy: %s", exc)
                self.client = None
        elif self.bearer_token:
            self.logger.warning("tweepy not available; Twitter sentiment disabled")

    def fetch(self) -> Optional[Dict[str, Any]]:
        if not self.client:
            return None
        try:
            max_results = min(max(self.max_tweets, 10), 100)
            response = self.client.search_recent_tweets(
                query=self.query,
                max_results=max_results,
                tweet_fields=["created_at", "public_metrics"]
            )
            tweets = response.data or []
            if not tweets:
                return None

            scores: List[float] = []
            for tweet in tweets:
                score = _score_text(tweet.text)
                if score is not None:
                    scores.append(score)

            if not scores:
                return None

            avg_score = sum(scores) / len(scores)
            confidence = min(len(scores) / max_results, 1.0)
            return {
                "sentiment": max(min(avg_score, 1.0), -1.0),
                "tweet_count": len(scores),
                "confidence": confidence
            }
        except Exception as exc:
            self.logger.warning("Twitter sentiment fetch failed: %s", exc)
            return None


class RedditSentimentClient:
    """Fetches recent Reddit sentiment using the PRAW."""

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.client_id = config.get("reddit_client_id") or os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = config.get("reddit_client_secret") or os.getenv("REDDIT_CLIENT_SECRET")
        self.user_agent = config.get("reddit_user_agent", "RenaissanceBot/1.0")
        self.subreddits = config.get("reddit_subreddits", "bitcoin,cryptocurrency,ethdev")
        
        self.reddit = None
        if REDDIT_AVAILABLE and self.client_id and self.client_secret:
            try:
                self.reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize Reddit client: {e}")

    def fetch(self) -> Optional[Dict[str, Any]]:
        if not self.reddit:
            return None
        try:
            scores = []
            for sub_name in self.subreddits.split(','):
                subreddit = self.reddit.subreddit(sub_name.strip())
                for submission in subreddit.hot(limit=10):
                    score = _score_text(submission.title + " " + submission.selftext)
                    if score is not None:
                        scores.append(score)
            
            if not scores:
                return None
                
            return {
                "sentiment": sum(scores) / len(scores),
                "count": len(scores),
                "confidence": min(len(scores) / 30, 1.0)
            }
        except Exception as e:
            self.logger.warning(f"Reddit sentiment fetch failed: {e}")
            return None


class NewsSentimentClient:
    """Fetches recent news sentiment using NewsAPI."""

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.api_key = config.get("newsapi_key") or os.getenv("NEWSAPI_KEY")
        self.query = config.get("news_query", "bitcoin OR ethereum OR cryptocurrency")
        
        self.client = None
        if NEWSAPI_AVAILABLE and self.api_key:
            try:
                self.client = NewsApiClient(api_key=self.api_key)
            except Exception as e:
                self.logger.warning(f"Failed to initialize NewsAPI client: {e}")

    def fetch(self) -> Optional[Dict[str, Any]]:
        if not self.client:
            return None
        try:
            response = self.client.get_everything(
                q=self.query,
                language='en',
                sort_by='publishedAt',
                page_size=20
            )
            
            articles = response.get('articles', [])
            if not articles:
                return None
                
            scores = []
            for article in articles:
                text = (article.get('title') or "") + " " + (article.get('description') or "")
                score = _score_text(text)
                if score is not None:
                    scores.append(score)
            
            if not scores:
                return None
                
            return {
                "sentiment": sum(scores) / len(scores),
                "count": len(scores),
                "confidence": min(len(scores) / 20, 1.0)
            }
        except Exception as e:
            self.logger.warning(f"NewsAPI sentiment fetch failed: {e}")
            return None


class AlternativeDataEngine:
    """Collects alternative data signals with graceful fallbacks."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}

        self.fear_greed_client = FearGreedClient(self.config.get("fear_greed", {}), logger=self.logger)
        self.twitter_client = TwitterSentimentClient(self.config.get("sentiment", {}), logger=self.logger)
        self.reddit_client = RedditSentimentClient(self.config.get("reddit", {}), logger=self.logger)
        self.news_client = NewsSentimentClient(self.config.get("news", {}), logger=self.logger)

    async def get_alternative_signals(self) -> AlternativeSignal:
        """Get alternative data signals asynchronously."""
        fear_task = asyncio.to_thread(self.fear_greed_client.fetch)
        twitter_task = asyncio.to_thread(self.twitter_client.fetch)
        reddit_task = asyncio.to_thread(self.reddit_client.fetch)
        news_task = asyncio.to_thread(self.news_client.fetch)

        fear_result, twitter_result, reddit_result, news_result = await asyncio.gather(
            fear_task, twitter_task, reddit_task, news_task
        )

        social_sentiment = 0.0
        reddit_sentiment = 0.0
        news_sentiment = 0.0
        market_psychology = 0.0
        on_chain_strength = 0.0
        confidence = 0.2

        confidences = []

        if twitter_result:
            social_sentiment = float(twitter_result.get("sentiment", 0.0))
            confidences.append(float(twitter_result.get("confidence", 0.0)))

        if reddit_result:
            reddit_sentiment = float(reddit_result.get("sentiment", 0.0))
            confidences.append(float(reddit_result.get("confidence", 0.0)))

        if news_result:
            news_sentiment = float(news_result.get("sentiment", 0.0))
            confidences.append(float(news_result.get("confidence", 0.0)))

        if fear_result:
            market_psychology = _normalize_fear_greed(fear_result.get("value", 50))
            confidences.append(0.7)

        if confidences:
            confidence = sum(confidences) / len(confidences)

        return AlternativeSignal(
            social_sentiment=social_sentiment,
            reddit_sentiment=reddit_sentiment,
            news_sentiment=news_sentiment,
            on_chain_strength=on_chain_strength,
            market_psychology=market_psychology,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc)
        )


def _score_text(text: str) -> Optional[float]:
    if not text:
        return None

    if SENTIMENT_BACKEND == "textblob" and TextBlob is not None:
        return float(TextBlob(text).sentiment.polarity)

    if VADER_AVAILABLE and SentimentIntensityAnalyzer is not None:
        analyzer = SentimentIntensityAnalyzer()
        return float(analyzer.polarity_scores(text).get("compound", 0.0))

    return None


def _normalize_fear_greed(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 50.0
    return max(min((numeric - 50.0) / 50.0, 1.0), -1.0)
