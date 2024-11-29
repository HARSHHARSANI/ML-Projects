import requests
import json

# Example API URL
url = "https://newsapi.org/v2/everything?q=tesla&from=2024-10-28&sortBy=publishedAt&apiKey=c8334439bd36423c89cb3157b3365a6f"

# Step 1: Make the API call
response = requests.get(url)

# Step 2: Check the response status
if response.status_code == 200:
    data = response.json()  # Parse the JSON response

    # Step 3: Extract the articles from the response
    articles = data.get("articles", [])

    # Step 4: Process each article into a structured format
    news_array = []
    for article in articles:
        news_item = {
            "source": article["source"].get("name", "Unknown"),
            "author": article.get("author", "Unknown"),
            "title": article.get("title", "No Title"),
            "description": article.get("description", "No Description"),
            "url": article.get("url", "No URL"),
            "urlToImage": article.get("urlToImage", "No Image"),
            "publishedAt": article.get("publishedAt", "No Date"),
            "content": article.get("content", "No Content")
        }
        news_array.append(news_item)

    # Print or save the array
    print(json.dumps(news_array, indent=2))  # Pretty print for clarity

else:
    print(f"Failed to fetch news. HTTP Status Code: {response.status_code}")
