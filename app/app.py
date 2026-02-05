from flask import Flask, request, jsonify, render_template_string
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import preprocess_text
from src.feature_extraction import extract_tfidf_features
from src.model_training import load_model, load_vectorizer

app = Flask(__name__)

# Load the trained model (assuming it's saved as model.pkl in models/)
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'model.pkl')
vectorizer_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'vectorizer.pkl')
if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    model = load_model('model.pkl')
    vectorizer = load_vectorizer('vectorizer.pkl')
else:
    model = None
    vectorizer = None

@app.route('/')
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sentiment Analysis</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                color: #333;
            }
            .container {
                background: white;
                padding: 40px;
                border-radius: 10px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                max-width: 600px;
                width: 100%;
                text-align: center;
            }
            h1 {
                color: #4a4a4a;
                margin-bottom: 30px;
                font-size: 2.5em;
            }
            label {
                display: block;
                margin-bottom: 10px;
                font-weight: bold;
                color: #555;
            }
            textarea {
                width: 100%;
                padding: 15px;
                border: 2px solid #ddd;
                border-radius: 5px;
                font-size: 16px;
                resize: vertical;
                box-sizing: border-box;
                margin-bottom: 20px;
            }
            textarea:focus {
                border-color: #667eea;
                outline: none;
            }
            input[type="submit"] {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 15px 30px;
                font-size: 18px;
                border-radius: 5px;
                cursor: pointer;
                transition: transform 0.2s;
            }
            input[type="submit"]:hover {
                transform: translateY(-2px);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 Sentiment Analysis</h1>
            <form action="/predict" method="post">
                <label for="review">Enter your review:</label>
                <textarea id="review" name="review" rows="6" placeholder="Type your review here..."></textarea>
                <input type="submit" value="Analyze Sentiment">
            </form>
        </div>
    </body>
    </html>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model not found. Please train the model first.'})

    review = request.form['review']
    processed_review = preprocess_text(review)
    features = vectorizer.transform([processed_review])
    prediction = model.predict(features)[0]
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    emoji = '😊' if sentiment == 'Positive' else '😞'
    color = '#4CAF50' if sentiment == 'Positive' else '#f44336'
    return render_template_string(f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sentiment Analysis Result</title>
        <style>
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(20px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                color: #333;
            }}
            .container {{
                background: white;
                padding: 50px;
                border-radius: 15px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.15);
                max-width: 700px;
                width: 100%;
                text-align: center;
                animation: fadeIn 1s ease-in;
                border: 2px solid #e1e8ed;
            }}
            h1 {{
                color: #2d3748;
                margin-bottom: 40px;
                font-size: 3em;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            }}
            .review {{
                background: #f7fafc;
                padding: 25px;
                border-radius: 10px;
                margin-bottom: 30px;
                font-style: italic;
                font-size: 1.2em;
                border-left: 5px solid {color};
                box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
            }}
            .sentiment {{
                font-size: 2.5em;
                font-weight: bold;
                color: {color};
                margin-bottom: 40px;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
            }}
            .emoji {{
                font-size: 4em;
                margin-bottom: 15px;
                animation: bounce 2s infinite;
            }}
            @keyframes bounce {{
                0%, 20%, 50%, 80%, 100% {{ transform: translateY(0); }}
                40% {{ transform: translateY(-10px); }}
                60% {{ transform: translateY(-5px); }}
            }}
            a {{
                display: inline-block;
                background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
                color: white;
                text-decoration: none;
                padding: 18px 35px;
                border-radius: 8px;
                transition: all 0.3s ease;
                font-size: 1.1em;
                font-weight: bold;
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }}
            a:hover {{
                transform: translateY(-3px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.3);
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 Sentiment Analysis Result</h1>
            <div class="review">"{review}"</div>
            <div class="emoji">{emoji}</div>
            <div class="sentiment">{sentiment}</div>
            <a href="/">Analyze Another Review</a>
        </div>
    </body>
    </html>
    ''')

if __name__ == '__main__':
    app.run(debug=True)
