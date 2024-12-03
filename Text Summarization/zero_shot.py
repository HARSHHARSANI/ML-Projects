import matplotlib

matplotlib.use('Agg')
import base64
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, Response
from transformers import pipeline
import torch
import mysql.connector  # Import mysql.connector
from mysql.connector import Error  # For error handling

app = Flask(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '12345678',
    'database': 'news_db'
}


def get_db_connection():
    """Establishes a connection to the MySQL database."""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None


model_name = "facebook/bart-large-mnli"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(device):
    """Loads the zero-shot classification model."""
    return pipeline("zero-shot-classification", model=model_name, device=device)


news_sentiment_analysis = load_model(device)
sentiment_list = ["positive", "negative", "neutral", "joy", "sadness", "anger", "fear", "trust"]


def generate_plot(article_str):
    """Generates a sentiment analysis plot for a given article string."""
    try:
        analysis_output = news_sentiment_analysis(
            article_str,
            candidate_labels=sentiment_list,
            multi_label=True
        )

        print("Reached here 1")
        sentiments = {emotion: score for emotion, score in zip(analysis_output['labels'], analysis_output['scores'])}

        df_sentiments = pd.DataFrame(sentiments.items(), columns=['Emotion', 'Score'])

        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        sns.set_palette("husl")

        print("Reached here 2")

        bar_plot = sns.barplot(x='Emotion', y='Score', data=df_sentiments)

        plt.title('Sentiment Analysis', fontsize=16, fontweight='bold')
        plt.xlabel('Emotion', fontsize=14)
        plt.ylabel('Score', fontsize=14)

        print("Reached here 3")

        for index, row in df_sentiments.iterrows():
            plt.text(index, row['Score'], round(row['Score'], 2), color='black', ha="center")

        plt.xticks(rotation=45)
        sns.despine()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        print("Reached here 4")

        return base64.b64encode(buf.getvalue()).decode('ascii')
    except Exception as e:
        print(f"Error during plotting: {e}")
        return None


@app.route("/")
def index():
    """Main route that fetches latest news articles, generates plots, and renders the template."""
    connection = get_db_connection()
    if not connection:
        return "Failed to connect to the database.", 500

    try:
        cursor = connection.cursor(dictionary=True)
        query = "SELECT id, title, content FROM news ORDER BY id DESC LIMIT 1"  # Fetch last 10 articles
        cursor.execute(query)
        results = cursor.fetchall()

        plots = []
        for row in results:
            article_id = row.get("id")
            title = row.get("title", "No Title")
            article_str = row.get("content", "")

            if article_str:
                plot_data = generate_plot(article_str)
                plots.append({
                    'id': article_id,
                    'title': title,
                    'plot': plot_data
                })
            else:
                plots.append({
                    'id': article_id,
                    'title': title,
                    'plot': None
                })

        return render_template("index.html", plots=plots)
    except Exception as e:
        print(f"Error fetching data from MySQL: {e}")
        return "An error occurred while fetching data.", 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


if __name__ == "__main__":
    app.run(debug=True)
