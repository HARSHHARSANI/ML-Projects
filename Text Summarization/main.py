from flask import Flask, render_template, request, redirect, url_for
import mysql.connector
from flask import Flask, request, jsonify


app = Flask(__name__)

# Connect to MySQL database
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="12345678",
    database="news_db"  # Specify the database here
)


@app.route('/')
def index():
    try:
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM news")
        results = mycursor.fetchall()
        return render_template('index.html', news=results)
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return "An error occurred while fetching data."
    finally:
        mycursor.close()  # Close the cursor to free up resources


@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()  # Get JSON data from request
    text_to_summarize = data.get('text', '')  # Extract text from JSON

    # Here you would call your summarization API or logic
    # For demonstration, we'll just return a dummy summary.
    summary = text_to_summarize[:50] + '...'  # Mock summary (first 50 characters)

    return jsonify({'summary': summary})  # Return summary as JSON


if __name__ == '__main__':
    app.run(debug=True)
