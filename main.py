import os
import json
import pandas as pd
import numpy as np
import nltk
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from fpdf import FPDF
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans

# NLTK downloads
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')

# Paths
DATA_PATH = "data/customer_review.csv"
OUTPUT_DIR = "output"
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

# -----------------------
# LOAD DATA
# -----------------------
def load_data(path):
    try:
        df = pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        print("UTF-8 failed, loaded with Latin-1 encoding.")
        df = pd.read_csv(path, encoding='latin-1')

    required_cols = ['review_id', 'Comments', 'Rating', 'Date']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"CSV must have a column named '{col}'")

    df = df.dropna(subset=['Comments'])

    # Fix Rating column: extract numeric value
    df['Rating'] = df['Rating'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)

    print(f"Dataset loaded. Total reviews: {len(df)}")
    return df

# -----------------------
# CLEAN TEXT
# -----------------------
def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# -----------------------
# RULE-BASED SENTIMENT
# -----------------------
def add_rule_sentiment(df):
    sia = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df['Comments'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

    def map_sentiment(score):
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    df['sentiment_label'] = df['sentiment_score'].apply(map_sentiment)
    return df

# -----------------------
# ML SENTIMENT CLASSIFIER
# -----------------------
def train_sentiment_classifier(df):
    df['cleaned_text'] = df['Comments'].apply(clean_text)

    X = df['cleaned_text']
    y = df['sentiment_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=5000)

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)

    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": report
    }

    return clf, vectorizer, metrics

# -----------------------
# TOPIC EXTRACTION
# -----------------------
def extract_topics(df, n_clusters=5):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(df['cleaned_text'])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['topic'] = kmeans.fit_predict(X_vec)

    terms = vectorizer.get_feature_names_out()
    topics = {}

    for i in range(n_clusters):
        idxs = np.argsort(kmeans.cluster_centers_[i])[::-1][:10]
        topics[f"Topic_{i}"] = [terms[j] for j in idxs]

    return topics

# -----------------------
# INSIGHTS
# -----------------------
def generate_insights(df):
    from collections import Counter
    pos_words = " ".join(df[df['sentiment_label']=='Positive']['cleaned_text']).split()
    neg_words = " ".join(df[df['sentiment_label']=='Negative']['cleaned_text']).split()

    insights = {
        "top_positive_words": [w for w, _ in Counter(pos_words).most_common(5)],
        "top_negative_words": [w for w, _ in Counter(neg_words).most_common(5)],
        "sentiment_distribution": df['sentiment_label'].value_counts().to_dict(),
        "average_rating": float(df['Rating'].mean())
    }

    return insights

# -----------------------
# MAIN PIPELINE
# -----------------------
def main():
    print("Starting Customer Review Analysis...")

    df = load_data(DATA_PATH)
    df = add_rule_sentiment(df)

    clf, vectorizer, sentiment_metrics = train_sentiment_classifier(df)
    topics = extract_topics(df)
    insights = generate_insights(df)

    # ******** SAVE FINAL JSON FOR FRONTEND ********
    final_output = {
        "sentiment_metrics": sentiment_metrics,
        "topics": topics,
        "insights": insights
    }

    with open(os.path.join(OUTPUT_DIR, "output.json"), "w") as f:
        json.dump(final_output, f, indent=4)

    print("Pipeline completed!")
    print("JSON saved at: output/output.json")

if __name__ == "__main__":
    main()
