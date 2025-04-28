import pytest
from unittest.mock import patch, MagicMock
from src.data_collection.fetch_data import DataFetcher
import requests
import tweepy

# Fixture to mock environment variables
@pytest.fixture
def mock_env_vars(monkeypatch):
    """
    Mocks environment variables for API keys.
    """
    monkeypatch.setenv("NEWSAPI_KEY", "test_news_api_key")
    monkeypatch.setenv("X_API_BEARER_TOKEN", "test_x_bearer_token")
    monkeypatch.setenv("FACEBOOK_ACCESS_TOKEN", "test_facebook_access_token")


@pytest.fixture
def data_fetcher(mock_env_vars):
    """
    Fixture to create a DataFetcher instance with mocked environment variables.
    """
    return DataFetcher()


@pytest.fixture
def mock_newsapi_response():
    """
    Fixture to mock a successful response from NewsAPI.
    """
    return {
        "status": "ok", "totalResults": 1, "articles": [
            {"source": {"id": "test-source", "name": "Test Source"},
             "author": "Test Author", "title": "Test Title",
             "description": "Test Description", "url": "http://testurl.com",
             "urlToImage": "http://testurl.com/image.jpg",
             "publishedAt": "2025-01-01T00:00:00Z", "content": "Test Content"}]
    }


@pytest.fixture
def mock_x_response():
    """
    Fixture to mock a successful response from X API.
    """
    return {"data": [{"id": "1234567890", "text": "Test Tweet Text"}], "meta": {"result_count": 1}}


@pytest.fixture
def mock_facebook_response():
    """
    Fixture to mock a successful response from Facebook API.
    """
    return [{"message": "Test Facebook Post", "id": "fb_post_1"}]


@patch("src.data_collection.fetch_data.requests.get")
def test_fetch_news(mock_get, data_fetcher, mock_newsapi_response):
    """
    Tests fetching news data from NewsAPI.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_newsapi_response
    mock_get.return_value = mock_response

    data = data_fetcher.fetch_news("test_query")
    assert len(data["articles"]) == 1
    assert data["articles"][0]["title"] == "Test Title"


@patch.object(tweepy.Client, "search_recent_tweets")
def test_fetch_tweets(mock_search_recent_tweets, data_fetcher, mock_x_response):
    """
    Tests fetching tweets from X.
    """
    # Mocking the return value of search_recent_tweets method
    mock_search_recent_tweets.return_value = MagicMock(data=mock_x_response["data"], meta=mock_x_response["meta"])

    tweets = data_fetcher.fetch_tweets("test_query")
    assert len(tweets.data) == 1
    assert tweets.data[0].text == "Test Tweet Text"


@patch("src.data_collection.fetch_data.facebook.GraphAPI.request")
def test_fetch_facebook_posts(mock_fb_request, data_fetcher, mock_facebook_response):
    """
    Tests fetching Facebook posts.
    """
    mock_fb_request.return_value = {"data": mock_facebook_response}
    posts = data_fetcher.fetch_facebook_posts("test_query")

    assert len(posts) == 1
    assert posts[0]["message"] == "Test Facebook Post"


@patch.object(requests, "get")
def test_fetch_news_api_error(mock_get, data_fetcher):
    """
    Tests handling of API errors when fetching news.
    """
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_get.return_value = mock_response
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "401 Client Error: Unauthorized for url: test_url"
    )

    data = data_fetcher.fetch_news("test_query")
    assert data is None