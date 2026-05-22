from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import json
import os
import re 
from utils import load_models, vectorize_text
from difflib import SequenceMatcher
import logging
import warnings
import webbrowser
from sklearn.exceptions import InconsistentVersionWarning
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


from nltk.stem import PorterStemmer
stemmer = PorterStemmer()


# Setup
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('gensim').setLevel(logging.ERROR)
logging.getLogger('smart_open').setLevel(logging.ERROR)
#logging.getLogger('werkzeug').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with something secure in production

# Load dataset
DATA_PATH = os.path.join('data', 'assignment3_II.csv')
print("Loading dataset...")
df = pd.read_csv(DATA_PATH).fillna("")
print("Dataset loaded.")

print("Flask app setup complete. Starting server...")
df_unique = df.drop_duplicates(subset='Clothing ID')
df_unique = df_unique.drop_duplicates(subset='Clothes Title')


def stem_text(text):
    return ' '.join([stemmer.stem(word) for word in text.split()])

# Precompute stemmed versions ONCE at startup
print("Precomputing stemmed columns...")
df_unique['Stemmed Title'] = df_unique['Clothes Title'].str.lower().apply(stem_text)
df_unique['Stemmed Desc'] = df_unique['Clothes Description'].str.lower().apply(stem_text)
print("Stemming complete.")


# Assign simplified categories based on title
def classify_category(title):
    title = title.lower()
    if 'dress' in title:
        return 'Dresses'
    elif 'top' in title:
        return 'Tops'
    elif 'pant' in title or 'jean' in title or 'trouser' in title:
        return 'Pants'
    elif 'jacket' in title or 'coat' in title:
        return 'Outerwear'
    elif 'skirt' in title:
        return 'Skirts'
    elif 'short' in title:
        return 'Shorts'
    else:
        return 'Others'

df_unique['Category'] = df_unique['Clothes Title'].apply(classify_category)

# Review file
REVIEWS_FILE = 'data/reviews.json'

# Load model components
print("Loading models...")
ft_model, tfidf_vectorizer, lr_model = load_models()
print("Models loaded.")

# In-memory reviews DB
reviews_db = {}

# Fuzzy search logic
def fuzzy_match(a, b, threshold=0.6):
    return SequenceMatcher(None, a, b).ratio() > threshold

@app.route('/')
def index():
    keyword = request.args.get('search', '').lower().strip()
    selected_category = request.args.get('category', 'All')

    if keyword:
        stemmed_query = stem_text(keyword)

        filtered = df_unique[
            (
                df_unique['Stemmed Title'].str.contains(stemmed_query, na=False) |
                df_unique['Stemmed Desc'].str.contains(stemmed_query, na=False)
            )
        ]

        if selected_category != "All":
            filtered = filtered[filtered['Category'] == selected_category]
    else:
        if selected_category == "All":
            filtered = df_unique
        else:
            filtered = df_unique[df_unique['Category'] == selected_category]

    titles = df_unique['Clothes Title'].dropna().unique().tolist()
    categories = ['All'] + sorted(df_unique['Category'].dropna().unique().tolist())

    return render_template(
        'index.html',
        items=filtered.to_dict(orient='records'),
        titles=titles,
        categories=categories,
        selected_category=selected_category
    )

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/item/<int:item_id>')
def item_detail(item_id):
    item = df[df['Clothing ID'] == item_id].iloc[0].to_dict()

    original_reviews = df[df['Clothing ID'] == item_id][['Title', 'Review Text', 'Rating', 'Recommended IND']].fillna("").to_dict(orient='records')
    user_reviews = reviews_db.get(item_id, [])
    all_reviews = user_reviews + original_reviews

    return render_template('item.html', item=item, reviews=all_reviews)

@app.route('/item/<int:item_id>/review', methods=['GET', 'POST'])
def new_review(item_id):
    item = df[df['Clothing ID'] == item_id].iloc[0].to_dict()

    if request.method == 'POST':
        title = request.form['title']
        review = request.form['review']
        rating = int(request.form['rating'])  # Still stored, not used in prediction

        # Only use title and review for prediction
        full_text = f"{title} {review}"
        vector = vectorize_text(full_text, ft_model, tfidf_vectorizer)
        prediction = int(lr_model.predict(vector)[0])

        return render_template(
            'confirm_review.html',
            item=item,
            title=title,
            review=review,
            rating=rating,
            predicted=prediction
        )

    return render_template('new_review.html', item=item)

@app.route('/item/<int:item_id>/review/confirm', methods=['POST'])
def submit_final_review(item_id):
    title = request.form['title']
    review = request.form['review']
    rating = int(request.form['rating'])
    predicted = int(request.form['predicted'])

    override = request.form['override']
    final_label = predicted if override == 'predicted' else int(override)

    new_entry = {
        "Title": title,
        "Review Text": review,
        "Rating": rating,
        "Recommended IND": final_label
    }

    reviews_db.setdefault(item_id, []).append(new_entry)

    with open(REVIEWS_FILE, 'w') as f:
        json.dump(reviews_db, f, indent=2)

    flash("Your review was submitted successfully. Thank you!")
    return redirect(url_for('item_detail', item_id=item_id))

from werkzeug.serving import run_simple

if __name__ == '__main__':
    print("Launching Flask...")
    webbrowser.open("http://127.0.0.1:5000")
    run_simple('127.0.0.1', 5000, app, use_debugger=True, use_reloader=False)


