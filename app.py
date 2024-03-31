from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sentence_transformers import util
import json
import pickle
from pymongo import MongoClient

app = Flask(__name__)

client = MongoClient('mongodb+srv://hj:hj@ask-krishna.ousbb2u.mongodb.net/?retryWrites=true&w=majority&appName=Ask-Krishna')
db = client['Ask-Krishna']
collection = db['Questions-Answers']

with open('trained_sloks_model.pickle', 'rb') as f:
    data = pickle.load(f)
    
model = data["model"]
trained_sloks = data["trained_sloks"]
meanings = data["meanings"]
adhyays = data["adhyays"]
verses_num = data["verses_num"]
sloks = data["sloks"]

@app.route('/api/ask_krishna', methods=['POST'])
def ask_krishna():
    if 'question' not in request.json:
        return jsonify({"error": "Question parameter is missing"}), 400
    
    question = request.json.get('question')
    
    new_question = model.encode(question)
    
    cosine_similarities = util.dot_score(new_question, trained_sloks)
    index = np.argmax(cosine_similarities)

    response = {
        "meaning": meanings[index],
        "slok": sloks[index],
        "adhyay": adhyays[index],
        "verse_num": verses_num[index]
    }
    
    document = {
        "question": question,
        "answer": response
    }
    
    try:
        collection.insert_one(document)
    except Exception as e:
        print(f"Error inserting document into MongoDB: {e}")

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
