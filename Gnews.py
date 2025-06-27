import os
import json
import logging
import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
import re
from datetime import datetime, timedelta
import argparse
from dateutil import parser

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
gnews_api_key = os.getenv("g-news_api_key")
if not openai_api_key or not gnews_api_key:
    raise RuntimeError("OPENAI_API_KEY or GNEWS_API_KEY not found in environment variables.")

# Initialize OpenAI client
try:
    openai_client = OpenAI(api_key=openai_api_key)
except OpenAIError as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    raise RuntimeError(f"Invalid OpenAI API key or configuration: {e}")

# Prompt and response cache
prompt_cache = {}
news_cache = {}
summary_cache = {}
conversation_history = {}

# Load the system prompt for summarization (cached)
def load_prompt(filepath="prompts.json", prompt_key="NEWS_SUMMARIZER"):
    cache_key = f"{filepath}:{prompt_key}"
    if cache_key in prompt_cache:
        return prompt_cache[cache_key]
    try:
        with open(filepath, "r") as file:
            data = json.load(file)
            prompt = data[prompt_key]
            prompt_cache[cache_key] = prompt
            return prompt
    except Exception as e:
        logger.error(f"Error loading prompt: {e}")
        raise RuntimeError(f"Error loading prompt: {e}")

# Sanitize input to remove potentially malicious content
def sanitize_input(text: str) -> str:
    """Removes special characters and excessive whitespace from input."""
    text = re.sub(r'[^\w\s.,!?]', '', text)  # Keep alphanumeric, spaces, and basic punctuation
    return ' '.join(text.split())  # Normalize whitespace

# Calculate relative timestamp
def get_relative_time(published_at):
    """Converts published_at timestamp to relative time (e.g., '2h ago'). Handles timezone-aware and naive datetimes."""
    pub_time = parser.parse(published_at)
    now = datetime.now(pub_time.tzinfo) if pub_time.tzinfo else datetime.now()
    time_diff = now - pub_time
    if time_diff < timedelta(minutes=60):
        minutes = int(time_diff.total_seconds() / 60)
        return f"{minutes}m ago"
    elif time_diff < timedelta(hours=24):
        hours = int(time_diff.total_seconds() / 3600)
        return f"{hours}h ago"
    else:
        days = int(time_diff.total_seconds() / 86400)
        return f"{days}d ago"

# Map topic to tag or detect article-level tag
def get_topic_tag(matched_topic, article_content):
    """Maps the matched topic to a tag or detects tag from article content."""
    topic_map = {
        "hr strategy and leadership": "Leadership",
        "workforce compliance and regulation": "Compliance",
        "talent acquisition and labor trends": "Talent",
        "compensation, benefits and rewards": "Compensation",
        "people development and culture": "Culture"
    }
    default_tag = topic_map.get(matched_topic, "General")
    
    # Basic article-level tag detection using OpenAI
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an HR expert. Assign one tag from [Leadership, Compliance, Talent, Compensation, Culture] based on the article content."},
                {"role": "user", "content": f"Analyze this content and assign a tag: {article_content[:500]}"}  # Limit to first 500 chars
            ],
            max_tokens=10
        )
        detected_tag = response.choices[0].message.content.strip()
        return detected_tag if detected_tag in topic_map.values() else default_tag
    except Exception as e:
        logger.warning(f"Failed to detect tag: {e}")
        return default_tag

# Fetch news from GNews API
@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=5))
def fetch_news(topic: str, max_results=5) -> list:
    """Fetches news articles from GNews API for a given topic, filters by HR domains, and uses the maximum lookback window (30 days)."""
    url = "https://gnews.io/api/v4/search"
    hr_domains = [
        "shrm.org",
        "hrdive.com",
        "fortune.com",
        "forbes.com"
    ]
    params = {
        "q": topic,
        "token": gnews_api_key,
        "lang": "en",
        "max": 50,  # fetch more to allow for filtering, GNews max is 100
        "sortby": "relevance",
        "in": "title,description,content",
        "from": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        # Filter for HR domains
        filtered = []
        seen_titles = set()
        for article in articles:
            url_val = article.get("url", "").lower()
            if any(domain in url_val for domain in hr_domains):
                title = article.get("title", "").lower()
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    filtered.append(article)
            if len(filtered) >= max_results:
                break
        if not filtered:
            logger.warning(f"No articles found for topic: {topic} in HR domains.")
        return filtered
    except Exception as e:
        logger.error(f"Error fetching news for topic {topic}: {e}")
        return []

# Summarize articles using OpenAI
@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=5))
def summarize_article(article: dict, prompt: str, model: str) -> str:
    """Summarizes a single article using OpenAI's GPT model with HR persona."""
    content = f"{article.get('title', '')}\n{article.get('description', '')}\n{article.get('content', '')}"
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Summarize the following news article in 2-3 sentences from an HR expert perspective:\n{content}"}
    ]
    try:
        # Force model to gpt-4-turbo unless explicitly overridden
        if model is None or model.strip() == '' or model == 'gpt-3.5-turbo':
            model = 'gpt-4-turbo'
        logger.info(f"Using OpenAI model: {model}")
        assistant_reply = ""
        response_stream = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=150,
            stream=True
        )
        for chunk in response_stream:
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    assistant_reply += content
        return assistant_reply.strip()
    except OpenAIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise

# Main backend-callable function
def get_news_summaries(user_input: str, user_id: str = None, prompt_file: str = "prompts.json", model: str = "gpt-4-turbo") -> dict:
    """
    Fetches and summarizes news articles based on user-specified topics.
    Caches results to save API costs and maintains conversation history.
    Returns a dict for structured backend integration.
    """
    if not user_input or not user_input.strip():
        logger.warning("Empty or invalid input received.")
        return {"status": "error", "message": "Please provide a valid topic or query."}

    # Sanitize input
    query = sanitize_input(user_input)
    if not query:
        logger.warning("Input is empty after sanitization.")
        return {"status": "error", "message": "Please provide a valid topic or query."}

    # Define allowed topics
    allowed_topics = [
        "hr strategy and leadership",
        "workforce compliance and regulation",
        "talent acquisition and labor trends",
        "compensation, benefits and rewards",
        "people development and culture"
    ]

    # Validate input against allowed topics
    matched_topic = next((topic for topic in allowed_topics if topic.lower() in query.lower()), None)
    if not matched_topic:
        logger.warning(f"Input does not match allowed topics: {query}")
        return {"status": "error", "message": f"Please specify one of: {', '.join(allowed_topics)}"}

    # Check cache for news articles
    cache_key = f"{matched_topic}:{datetime.now().strftime('%Y-%m-%d')}"
    if cache_key in news_cache:
        logger.info(f"Cache hit for news articles: {matched_topic}")
        articles = news_cache[cache_key]
        cached_articles = True
    else:
        articles = fetch_news(matched_topic)
        news_cache[cache_key] = articles
        cached_articles = False
        # Clear cache older than 1 day
        for key in list(news_cache.keys()):
            key_date = datetime.strptime(key.split(':')[-1], '%Y-%m-%d')
            if datetime.now() - key_date > timedelta(days=1):
                del news_cache[key]

    if not articles:
        return {"status": "error", "message": f"No articles found for topic: {matched_topic}"}

    # Load summarization prompt
    try:
        summary_prompt = load_prompt(prompt_file, "NEWS_SUMMARIZER")
    except RuntimeError:
        return {"status": "error", "message": "Failed to load summarization prompt."}

    # Summarize articles
    summaries = []
    for article in articles:
        article_key = f"{article.get('title', '')}:{matched_topic}"
        content = f"{article.get('title', '')}\n{article.get('description', '')}\n{article.get('content', '')}"
        if article_key in summary_cache:
            logger.info(f"Cache hit for article summary: {article.get('title', 'Unknown')}")
            summaries.append({
                "title": article.get("title", "No title"),
                "url": article.get("url", ""),
                "summary": summary_cache[article_key],
                "source": article.get("source", {}).get("name", "Unknown"),
                "published_at": article.get("publishedAt", datetime.now().isoformat()),
                "cached": True
            })
        else:
            try:
                summary = summarize_article(article, summary_prompt, model)
                summary_cache[article_key] = summary
                tag = get_topic_tag(matched_topic, content)
                summaries.append({
                    "title": article.get("title", "No title"),
                    "url": article.get("url", ""),
                    "summary": summary,
                    "source": article.get("source", {}).get("name", "Unknown"),
                    "published_at": article.get("publishedAt", datetime.now().isoformat()),
                    "tag": tag,
                    "cached": False
                })
            except Exception as e:
                logger.error(f"Error summarizing article {article.get('title', 'Unknown')}: {e}")
                continue

    if not summaries:
        return {"status": "error", "message": "Failed to generate summaries for articles."}

    # Update conversation history
    if user_id:
        if user_id not in conversation_history:
            conversation_history[user_id] = []
        conversation_history[user_id].append({
            "role": "user",
            "content": query,
            "timestamp": datetime.now().isoformat(),
            "topic": matched_topic,
            "summaries": summaries
        })
        conversation_history[user_id] = conversation_history[user_id][-10:]  # Keep last 10 interactions

    logger.info(f"Generated summaries for topic: {matched_topic}")
    return {
        "status": "success",
        "topic": matched_topic,
        "articles": summaries,
        "cached_articles": cached_articles,
        "total_articles": len(summaries)
    }

# CLI interface
def main():
    parser = argparse.ArgumentParser(description="Fetch and summarize news articles using GNews and OpenAI APIs.")
    parser.add_argument("--topic", required=True, help="Topic to fetch news for (e.g., 'hr strategy and leadership')")
    parser.add_argument("--user-id", help="User ID for conversation history (optional)", default=None)
    parser.add_argument("--prompt-file", help="Path to prompts JSON file", default="prompts.json")
    parser.add_argument("--model", help="OpenAI model to use (default: gpt-3.5-turbo)", default="gpt-3.5-turbo")
    parser.add_argument("--output-json", help="Save output to a JSON file", type=str, default=None)
    args = parser.parse_args()

    response = get_news_summaries(args.topic, args.user_id, args.prompt_file, args.model)
    if response["status"] == "success":
        print(f"\nTopic: {response['topic']}")
        print(f"Total Articles: {response['total_articles']}")
        print(f"Cached Articles: {response['cached_articles']}")
        print("\nArticles:")
        for i, article in enumerate(response["articles"], 1):
            timestamp = get_relative_time(article["published_at"])
            print(f"\n=== News Card {i} ===")
            print(f"Headline: {article['title']}")
            print(f"AI-generated Summary: {article['summary']}")
            print(f"Source: {article['source']}")
            print(f"Link to original article: {article['url']} (Read more)")
            print(f"Topic Tag: {article['tag']}")
            print(f"Timestamp: {timestamp}")
        print("\n--- Page Footer ---")
        print("Summaries are AI-generated. Full articles belong to their original publishers.")
        if args.output_json:
            try:
                with open(args.output_json, "w") as f:
                    json.dump(response, f, indent=4)
                print(f"\nOutput saved to {args.output_json}")
            except Exception as e:
                print(f"Error saving JSON output: {e}")
    else:
        print(f"Error: {response['message']}")

if __name__ == "__main__":
    main()