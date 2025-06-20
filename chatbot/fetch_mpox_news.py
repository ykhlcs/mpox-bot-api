from venv import logger
from newsapi import NewsApiClient
import os

newsapi = NewsApiClient(api_key=os.getenv("NEWSAPI_KEY"))

def fetch_monkeypox_news():
    try:
        top_headlines = newsapi.get_everything(
            q="monkeypox",
            language="en",
            sort_by="publishedAt",
            page_size=5  # adjust number as needed
        )
        return [(article['title'], article['url']) for article in top_headlines['articles']]
    except Exception as e:
        logger.error(f"NewsAPI error: {e}")
        return []
