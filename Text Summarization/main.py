from flask import Flask, render_template, request, jsonify, url_for
import mysql.connector
from transformers import pipeline
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses INFO and WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses INFO, WARNING, and ERROR logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

# Connect to MySQL database
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="12345678",
    database="news_db"  # Specify the database here
)

# Initialize Hugging Face summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def get_db_connection():
    if not mydb.is_connected():
        mydb.reconnect()
    return mydb


@app.route('/')
def index():
    mycursor = None
    try:
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM news ORDER BY id DESC LIMIT 25")
        results = mycursor.fetchall()
        return render_template('index.html', news=results)
    except mysql.connector.Error as err:
        print(f"Database Error: {err}")
        return "An error occurred while fetching data from the database.", 500
    finally:
        if mycursor:
            mycursor.close()  # Ensure cursor is closed only if initialized


@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text_to_summarize = data.get('text', '').strip()
    if not text_to_summarize:
        return jsonify({'error': 'No text provided to summarize.'}), 400

    # Step 1: Save full content to the database
    try:
        mycursor = mydb.cursor()
        save_query = "INSERT INTO summaries (full_text) VALUES (%s)"
        mycursor.execute(save_query, (text_to_summarize,))
        mydb.commit()
        saved_id = mycursor.lastrowid  # Get the ID of the saved record
    except mysql.connector.Error as err:
        print(f"Error saving full text: {err}")
        return jsonify({'error': 'Failed to save full text'}), 500
    finally:
        mycursor.close()

    # Step 2: Generate a summary using Hugging Face pipeline
    try:
        print(f"Summarizing: {text_to_summarize[:250]}...")  # Print a preview of the text
        summarized_text = summarizer(text_to_summarize, max_length=100, min_length=25, do_sample=False)[0][
            'summary_text']
    except Exception as e:
        print(f"Error in summarization: {e}")
        return jsonify({'error': 'Failed to generate summary'}), 500

    # Step 3: Save the summary back to the database
    try:
        mycursor = mydb.cursor()
        update_query = "UPDATE summaries SET summary = %s WHERE id = %s"
        mycursor.execute(update_query, (summarized_text, saved_id))
        mydb.commit()
    except mysql.connector.Error as err:
        print(f"Error saving summary: {err}")
        return jsonify({'error': 'Failed to save summary'}), 500
    finally:
        mycursor.close()

    # Step 4: Redirect to the summary page
    return jsonify({'redirect_url': url_for('show_summary', summary_id=saved_id)})


@app.route('/summary/<int:summary_id>')
def show_summary(summary_id):
    try:
        mycursor = mydb.cursor(dictionary=True)
        query = "SELECT full_text, summary FROM summaries WHERE id = %s"
        mycursor.execute(query, (summary_id,))
        result = mycursor.fetchone()
        if not result:
            return "Summary not found.", 404
        return render_template('summary.html', full_text=result['full_text'], summary=result['summary'])
    except mysql.connector.Error as err:
        print(f"Error fetching summary: {err}")
        return "An error occurred while fetching the summary.", 500
    finally:
        if mycursor:
            mycursor.close()


if __name__ == '__main__':
    app.run(debug=True)
