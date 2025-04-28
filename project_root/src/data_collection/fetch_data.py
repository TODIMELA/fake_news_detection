import requests
import json
import os
import logging
import facebook
from datetime import datetime, timedelta
import tweepy
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataFetcher:
    """
    A class for fetching data from NewsAPI, X, and Facebook.
    """
    def __init__(self):
        # Retrieve API keys and tokens from environment variables
        self.news_api_key = os.environ.get("NEWSAPI_KEY")
        self.x_api_bearer_token = os.environ.get("X_API_BEARER_TOKEN")
        self.facebook_access_token = os.environ.get("FACEBOOK_ACCESS_TOKEN")
        
        # Check if API keys are set
        if not all([self.news_api_key, self.x_api_bearer_token, self.facebook_access_token]):
            raise ValueError("One or more API keys are missing from environment variables.")

        self.news_api_url = "https://newsapi.org/v2/everything"
        self.data_dir = "data/raw"
        self.x_client = tweepy.Client(bearer_token=self.x_api_bearer_token)
        os.makedirs(self.data_dir, exist_ok=True)

    def fetch_news(self, query="AI", from_date=None, to_date=None, page=1, page_size=100):
        """Fetches news articles from NewsAPI."""
        headers = {"Authorization": f"Bearer {self.news_api_key}"}  # NewsAPI requires the API key in the header
        params = {
            "q": query,
            "from": from_date,
            "to": to_date,
            "page": page,
            "pageSize": page_size,
            
            "apiKey": self.news_api_key,  # NewsAPI requires the API key in the params
            "language": "en",
            "sortBy": "relevancy",  # Sort by relevancy
        }
        try:
            response = requests.get(self.news_api_url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching news: {e}")
            return None

    def fetch_tweets(self, query:str, max_results: int=100) :
        """Fetches tweets from X."""
        try:
            tweets = self.x_client.search_recent_tweets(query=query, max_results=max_results)
            return tweets
        except Exception as e:
            print(f"Error fetching tweets: {e}")
            logging.error(f"Error fetching tweets: {e}")
            return None

    def fetch_facebook_posts(self, query: str, limit: int = 100) -> List[dict]:
        """
        Fetches posts from Facebook based on a search query.
        Args:
            query: The search query string.
            limit: The maximum number of posts to fetch.
        Returns:
            A list of posts.
        """
        try:
            graph = facebook.GraphAPI(access_token=self.facebook_access_token)
            search_results = graph.request(
                "/search",
                {"q": query, "type": "post", "limit": limit, "fields": "message,id,created_time"},
            )
            return search_results.get("data", [])

        except facebook.GraphAPIError as e:
            print(f"Error fetching Facebook posts: {e}")
            return []

    def save_data(self, data:dict, filename: str) :
        """Saves data to a JSON file in the specified directory."""
        file_path = os.path.join(self.data_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def fetch_and_save_news(self, query="AI", days_ago=7):
        """Fetches and saves news articles for a given query and date range."""
        logging.info(f"Fetching and saving news for query: {query}")
        today = datetime.now()
        from_date = (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        to_date = today.strftime("%Y-%m-%d")

        page = 1
        all_articles = []
        while True:
            news_data = self.fetch_news(query, from_date, to_date, page=page)  # Fetch news articles
            if not news_data or not news_data.get("articles"):
                break
            all_articles.extend(news_data["articles"])  # Add the articles to the list
            page += 1  # Increment page number for next request
            if page >= 5:
                break

        if all_articles:
            filename = f"news_{query}_{from_date}_{to_date}.json"
            self.save_data(all_articles, filename)
            logging.info(f"Saved {len(all_articles)} news articles to {filename}")

    def fetch_and_save_tweets(self, query, max_results=100):
        """Fetches and saves tweets for a given query."""
        logging.info(f"Fetching and saving tweets for query: {query}")
        tweets_data = self.fetch_tweets(query, max_results)
        
        if tweets_data and tweets_data.data:
            tweets_list = []
            for tweet in tweets_data.data:
                tweets_list.append(tweet.text)
            filename = f"tweets_{query}.json"
            self.save_data(tweets_list, filename)
            print(f"Saved {len(tweets_list)} tweets to {filename}")
            logging.info(f"Saved {len(tweets_list)} tweets to {filename}")
        else:
            logging.warning("No Tweets found")

    def fetch_and_save_facebook_posts(self, query: str, limit: int = 100):
        """
        Fetches and saves Facebook posts based on a search query.
        """
        logging.info(f"Fetching and saving Facebook posts for query: {query}")
        posts = self.fetch_facebook_posts(query, limit)
        if posts:
            filename = f"facebook_{query}.json"
            self.save_data(posts, filename)
            logging.info(f"Saved {len(posts)} Facebook posts to {filename}")


if __name__ == "__main__":
    fetcher = DataFetcher()
    fetcher.fetch_and_save_facebook_posts("AI news", limit=50)
    fetcher.fetch_and_save_news()
    fetcher.fetch_and_save_tweets(query="AI news")
