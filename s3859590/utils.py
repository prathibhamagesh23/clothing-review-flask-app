import os
import numpy as np
from gensim.models import FastText
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Paths to your model files
MODEL_DIR = 'models'
FT_PATH = os.path.join(MODEL_DIR, 'fasttext_model.bin')
LR_PATH = os.path.join(MODEL_DIR, 'logistic_regression_model.pkl')
TFIDF_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')

# Load models
def load_models():
    ft_model = FastText.load(FT_PATH)
    with open(LR_PATH, 'rb') as f:
        lr_model = pickle.load(f)
    with open(TFIDF_PATH, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    return ft_model, tfidf_vectorizer, lr_model

# Vectorize text using FastText + TF-IDF
def vectorize_text(text, model, vectorizer):
    words = text.split()
    tfidf_weights = vectorizer.transform([text]).toarray()[0]
    feature_names = vectorizer.get_feature_names_out()
    weighted_vectors = []

    for word in words:
        if word in feature_names:
            try:
                weight = tfidf_weights[feature_names.tolist().index(word)]
                word_vec = model.wv[word]
                weighted_vectors.append(word_vec * weight)
            except KeyError:
                continue

    if weighted_vectors:
        return np.mean(weighted_vectors, axis=0).reshape(1, -1)
    else:
        return np.zeros((1, model.vector_size))
