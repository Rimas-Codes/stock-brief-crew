import yfinance as yf
from crewai.tools import tool

@tool('Get Stock Data')
def get_stock_data(ticker: str) -> str:
    """Get current price and recent news for a given stock ticker symbol."""
    stock = yf.Ticker(ticker)

    info = stock.info
    current_price = info.get('currentPrice', 'N/A')
    previous_close = info.get('previousClose', 'N/A')

    if current_price != 'N/A' and previous_close != 'N/A':
        pct_change = ((current_price - previous_close)/previous_close) * 100
        price_info = f'Price {current_price:.2f} ({pct_change:+.2f}% vs previous close {previous_close:.2f})'
    else:
        price_info = 'No price info available'

    news = stock.news
    headlines = []

    for news_item in news[:8]:
        content = news_item.get("content") or {}
        title = (content.get("title") or "").strip()
        published = str(content.get("pubDate") or "")[:10] or "date unavailable"
        link_data = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
        if isinstance(link_data, dict):
            link = link_data.get("url") or ""
        elif isinstance(link_data, str):
            link = link_data
        else:
            link = ""

        if title:
            headline = f"- [{published}] {title}"
            if link:
                headline += f" ({link})"
            headlines.append(headline)

    if not headlines:
        headlines = ['- No headlines available.']

    news_text = '\n'.join(headlines)

    return f'{price_info}\n\nRecent News:\n{news_text}'