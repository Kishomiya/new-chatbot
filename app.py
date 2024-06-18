from flask import Flask, render_template, request, jsonify
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

app = Flask(__name__)

# Load trained model and tokenizer
model = load_model('chatbot_model.keras')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('label_encoder.pickle', 'rb') as handle:
    lbl_encoder = pickle.load(handle)

# Load intents file
with open('intents.json', encoding='utf-8') as file:
    data = json.load(file)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Set up NLTK stopwords
stop_words = set(stopwords.words('english'))

def predict_intent(message):
    # Tokenize the message
    tokens = word_tokenize(message)
    
    # Remove stopwords
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # Check if any keywords match intent patterns
    for intent in data['intents']:
        for pattern in intent['patterns']:
            if any(keyword.lower() in pattern.lower() for keyword in filtered_tokens):
                return random.choice(intent['responses'])
    
    # If message not found in intents, return a default response
    return "This is the Social Media chatbot. Ask anything related to social Media."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get', methods=['POST'])
def get_bot_response():
    user_text = request.json['message']
    
    # Check for empty message
    if not user_text.strip():
        return jsonify({'reply': "Please enter a message."})

    # Get response based on intent prediction
    bot_response = predict_intent(user_text)

    return jsonify({'reply': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
