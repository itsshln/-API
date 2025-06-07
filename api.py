from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import re
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

app = Flask(__name__)

# Загрузка модели
model = load_model('text_clf.keras')

# Загрузка и инициализация необходимых компонентов
nltk.download('wordnet')
nltk.download('punkt')

lemmatizer = WordNetLemmatizer()

# Функции предобработки текста
def normal_text(text):
    if isinstance(text, str):
        text = re.sub(r'\W+', ' ', text)
        text = text.lower()
        words = word_tokenize(text)
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)
    else:
        return ''

try:
    with open('tfidf_vectorizer.joblib', 'rb') as f:
        tfidf_vectorizer = joblib.load(f)
except FileNotFoundError:
    raise Exception("Файл 'tfidf_vectorizer.joblib' не найден. Убедитесь, что вы сохранили обученный векторизатор.")

# Классы для классификации
CLASSES = ['cited', 'applied', 'followed', 'referred to', 'related',
           'considered', 'discussed', 'distinguished', 'affirmed', 'approved']

# Инициализация LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(CLASSES)  # Фитируем на всех возможных классах

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    
    # Предобработка текста
    processed_text = normal_text(text)
    
    # Векторизация текста (предполагаем, что tfidf_vectorizer уже обучен и сохранен)
    # На практике вам нужно сохранить и загрузить обученный векторизатор
    vectorized = tfidf_vectorizer.transform([processed_text]).toarray()
    
    # Предсказание
    prediction = model.predict(vectorized)[0]
    predicted_class_index = np.argmax(prediction)
    predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
    
    # Формирование вероятностей для всех классов
    probabilities = {cls: float(prediction[i]) for i, cls in enumerate(CLASSES)}
    
    return jsonify({
        'text': text,
        'case_outcome': predicted_class,
        'confidence': float(np.max(prediction)),
        'probabilities': probabilities
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)