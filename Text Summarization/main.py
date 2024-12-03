import pandas as pd
import matplotlib
import base64
from io import BytesIO
import os
from flask import Flask, render_template, request, jsonify, url_for
import mysql.connector
from transformers import pipeline
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Configure Matplotlib for non-GUI environments
matplotlib.use('Agg')

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Flask app
app = Flask(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '12345678',
    'database': 'news_db',
}


# Establish database connection
def get_db_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
    except mysql.connector.Error as e:
        print(f"Database connection error: {e}")
        return None


# Load models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sentiment_analyzer = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

# Sentiment labels
SENTIMENT_LIST = ["positive", "negative", "neutral", "joy", "sadness", "anger", "fear", "trust"]


# Function to generate sentiment analysis plot and save to CSV
def generate_sentiment_plot_and_save_to_csv(text, summary):
    try:
        # Perform sentiment analysis
        analysis_output = sentiment_analyzer(text, candidate_labels=SENTIMENT_LIST, multi_label=True)
        sentiments = {emotion: score for emotion, score in zip(analysis_output['labels'], analysis_output['scores'])}

        # Generate the sentiment plot
        df_sentiments = pd.DataFrame(sentiments.items(), columns=['Emotion', 'Score'])

        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        sns.set_palette("husl")
        sns.barplot(x='Emotion', y='Score', data=df_sentiments)

        plt.title('Sentiment Analysis', fontsize=16, fontweight='bold')
        plt.xlabel('Emotion', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.xticks(rotation=45)
        sns.despine()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        plot_base64 = base64.b64encode(buf.getvalue()).decode('ascii')

        # Prepare data for CSV
        sentiment_data = {
            "Text Summary": summary,
            "Positive Score": sentiments.get("positive", 0),
            "Negative Score": sentiments.get("negative", 0),
            "Neutral Score": sentiments.get("neutral", 0),
            "Joy Score": sentiments.get("joy", 0),
            "Sadness Score": sentiments.get("sadness", 0),
            "Anger Score": sentiments.get("anger", 0),
            "Fear Score": sentiments.get("fear", 0),
            "Trust Score": sentiments.get("trust", 0),
            "Other Emotions": str(sentiments)  # Saving all emotions as a string
        }

        # Save to CSV
        sentiment_df = pd.DataFrame([sentiment_data])

        # If the CSV file doesn't exist, create it with header
        file_exists = os.path.exists("sentiment_results.csv")
        sentiment_df.to_csv("sentiment_results.csv", mode='a', header=not file_exists, index=False)

        return plot_base64
    except Exception as e:
        print(f"Error generating sentiment plot and saving to CSV: {e}")
        return None


@app.route('/')
def index():
    connection = get_db_connection()
    if not connection:
        return "Failed to connect to the database.", 500

    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT id, title, content FROM news ORDER BY id DESC LIMIT 10")
        results = cursor.fetchall()
        return render_template('index.html', news=results)
    except mysql.connector.Error as e:
        print(f"Database error: {e}")
        return "An error occurred while fetching data.", 500
    finally:
        cursor.close()
        connection.close()


@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text_to_summarize = data.get('text', '').strip()
    if not text_to_summarize:
        return jsonify({'error': 'No text provided to summarize.'}), 400

    # Save full content to database
    connection = get_db_connection()
    if not connection:
        return jsonify({'error': 'Database connection failed.'}), 500
    try:
        cursor = connection.cursor()
        save_query = "INSERT INTO summaries (full_text) VALUES (%s)"
        cursor.execute(save_query, (text_to_summarize,))
        connection.commit()
        saved_id = cursor.lastrowid
    except mysql.connector.Error as e:
        print(f"Error saving full text: {e}")
        return jsonify({'error': 'Failed to save full text'}), 500
    finally:
        cursor.close()

    # Generate summary
    try:
        summarized_text = summarizer(text_to_summarize, max_length=100, min_length=25, do_sample=False)[0][
            'summary_text']
    except Exception as e:
        print(f"Summarization error: {e}")
        return jsonify({'error': 'Failed to generate summary'}), 500

    # Save summary to database
    try:
        cursor = connection.cursor()
        update_query = "UPDATE summaries SET summary = %s WHERE id = %s"
        cursor.execute(update_query, (summarized_text, saved_id))
        connection.commit()
    except mysql.connector.Error as e:
        print(f"Error saving summary: {e}")
        return jsonify({'error': 'Failed to save summary'}), 500
    finally:
        cursor.close()

    # Generate sentiment plot and save to CSV
    plot_data = generate_sentiment_plot_and_save_to_csv(summarized_text, summarized_text)

    # Render summary page
    return jsonify({
        'redirect_url': url_for('show_summary', summary_id=saved_id),
        'plot': plot_data
    })


@app.route('/summary/<int:summary_id>')
def show_summary(summary_id):
    connection = get_db_connection()
    if not connection:
        return "Failed to connect to the database.", 500

    try:
        cursor = connection.cursor(dictionary=True)
        query = "SELECT full_text, summary FROM summaries WHERE id = %s"
        cursor.execute(query, (summary_id,))
        result = cursor.fetchone()
        if not result:
            return "Summary not found.", 404

        plot_data = generate_sentiment_plot_and_save_to_csv(result['summary'], result['summary'])
        return render_template('summary.html', full_text=result['full_text'], summary=result['summary'], plot=plot_data)
    except mysql.connector.Error as e:
        print(f"Error fetching summary: {e}")
        return "An error occurred while fetching the summary.", 500
    finally:
        cursor.close()
        connection.close()


if __name__ == '__main__':
    app.run(debug=True)
