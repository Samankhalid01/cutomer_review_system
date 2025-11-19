#!/usr/bin/env python3
"""
main.py
End-to-end Customer Review Intelligence System

Place dataset at: project/data/customer_reviews.csv
Columns required: review_id, review_text, rating (1-5), date

Run: python main.py
"""

import os
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.pipeline import make_pipeline
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet

# -----------------------------
# 0. Setup / Config
# -----------------------------
DATA_PATH = "data/customer_reviews.csv"
OUTPUT_DIR = "output"
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")
MODELS_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# NLTK downloads (if not present)
nltk_data_needed = ["stopwords", "punkt", "wordnet", "omw-1.4"]
for pkg in nltk_data_needed:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except Exception:
        nltk.download(pkg)

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

RANDOM_STATE = 42

# -----------------------------
# Utilities
# -----------------------------
def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# -----------------------------
# Part 1: Data Ingestion
# -----------------------------
def load_and_clean_csv(path):
    df = pd.read_csv(path)
    # Ensure required columns exist
    required = {"review_id", "review_text", "rating", "date"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must include columns: {required}. Found: {set(df.columns)}")
    # Drop NaN reviews
    df = df.dropna(subset=["review_text"]).copy()
    # Convert types
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Drop rows with invalid dates (optional)
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    # Basic stats
    stats = {
        "total_reviews": int(len(df)),
        "average_rating": float(df["rating"].mean()),
        "min_date": str(df["date"].min().date()),
        "max_date": str(df["date"].max().date())
    }
    save_json(stats, os.path.join(OUTPUT_DIR, "data_summary.json"))
    print("Loaded dataset. Stats saved to output/data_summary.json")
    return df

# -----------------------------
# Part 2: Text Preprocessing
# -----------------------------
def clean_text(text: str) -> str:
    # Lowercase
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    # Remove non-alphanumeric (keep spaces)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Tokenize (simple split)
    tokens = text.split()
    # Remove stopwords and short tokens
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    # Lemmatize
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# -----------------------------
# Part 3A: Rule-based sentiment
# -----------------------------
POS_KEYWORDS = {"good","great","excellent","love","loved","awesome","amazing","perfect","nice","satisfied","fast","happy"}
NEG_KEYWORDS = {"bad","terrible","awful","worst","hate","hated","slow","poor","disappoint","problem","issue","broken","refund"}

def rule_based_sentiment(text: str) -> str:
    # Count positive & negative keyword occurrences
    tokens = text.split()
    pos = sum(1 for t in tokens if t in POS_KEYWORDS)
    neg = sum(1 for t in tokens if t in NEG_KEYWORDS)
    if pos > neg and pos > 0:
        return "positive"
    elif neg > pos and neg > 0:
        return "negative"
    else:
        return "neutral"

# -----------------------------
# Part 3B: ML sentiment classifier (train)
# -----------------------------
def train_sentiment_classifier(df, test_size=0.2):
    # Create target labels from rating: (1-2) negative, (3) neutral, (4-5) positive
    def rating_to_label(r):
        r = float(r)
        if r <= 2.0:
            return "negative"
        elif r == 3.0:
            return "neutral"
        else:
            return "positive"
    df = df.copy()
    df["label"] = df["rating"].apply(rating_to_label)
    # Use cleaned text
    X = df["cleaned_text"].values
    y = df["label"].values
    # TF-IDF vectorizer
    vect = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    X_tfidf = vect.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    # Train Logistic Regression
    clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    # Evaluate
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="weighted", zero_division=0)
    metrics = {
        "accuracy": float(acc),
        "precision_weighted": float(prec),
        "recall_weighted": float(rec),
        "f1_weighted": float(f1),
        "classification_report": classification_report(y_test, preds, zero_division=0, output_dict=True)
    }
    # Save model and vectorizer together
    joblib.dump({"vectorizer": vect, "classifier": clf}, os.path.join(MODELS_DIR, "sentiment_model.pkl"))
    save_json(metrics, os.path.join(OUTPUT_DIR, "sentiment_metrics.json"))
    print("Trained sentiment model saved to models/sentiment_model.pkl")
    print("Sentiment metrics saved to output/sentiment_metrics.json")
    return vect, clf, metrics

# -----------------------------
# Part 4: Topic Extraction (KMeans)
# -----------------------------
def extract_topics(df, n_clusters=6, top_n_terms=10):
    vect = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X = vect.fit_transform(df["cleaned_text"].values)
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X)
    # Top terms per cluster
    terms = np.array(vect.get_feature_names_out())
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    topics = {}
    for i in range(n_clusters):
        top_terms = [terms[ind] for ind in order_centroids[i, :top_n_terms]]
        topics[f"topic_{i}"] = top_terms
    # Attach label to df
    df_topics = df.copy()
    df_topics["topic"] = labels
    save_json(topics, os.path.join(OUTPUT_DIR, "topics.json"))
    print(f"Extracted {n_clusters} topics. Saved to output/topics.json")
    return df_topics, topics, kmeans, vect

# -----------------------------
# Part 5: Insights
# -----------------------------
def generate_insights(df, topics):
    # Top positive & negative keywords via rule keywords frequency in respective sets
    pos_text = " ".join(df[df["rule_sentiment"]=="positive"]["cleaned_text"].values)
    neg_text = " ".join(df[df["rule_sentiment"]=="negative"]["cleaned_text"].values)
    pos_counts = Counter(pos_text.split()).most_common(20)
    neg_counts = Counter(neg_text.split()).most_common(20)
    # Most common complaints = negative top tokens
    insights = {
        "top_positive_keywords": [w for w,_ in pos_counts[:5]],
        "top_negative_keywords": [w for w,_ in neg_counts[:5]],
    }
    # Average sentiment score per month (map positive=1 neutral=0 negative=-1)
    mapping = {"positive":1,"neutral":0,"negative":-1}
    df["sentiment_score"] = df["rule_sentiment"].map(mapping)
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    monthly = df.groupby("year_month")["sentiment_score"].mean().reset_index().to_dict(orient="records")
    insights["monthly_sentiment"] = monthly
    # Sentiment distribution
    sentiment_counts = df["rule_sentiment"].value_counts().to_dict()
    insights["sentiment_distribution"] = sentiment_counts
    save_json(insights, os.path.join(OUTPUT_DIR, "insights.json"))
    print("Insights saved to output/insights.json")
    return insights

# -----------------------------
# Part 6: Visualizations
# -----------------------------
def make_charts(df, topics_df):
    # 1. Sentiment distribution bar chart (rule-based)
    plt.figure(figsize=(6,4))
    sns.countplot(x="rule_sentiment", data=df, order=["positive","neutral","negative"])
    plt.title("Sentiment Distribution (Rule-Based)")
    plt.savefig(os.path.join(CHARTS_DIR, "sentiment_distribution.png"), bbox_inches="tight")
    plt.close()

    # 2. Ratings vs sentiment
    plt.figure(figsize=(6,4))
    sns.boxplot(x="rule_sentiment", y="rating", data=df, order=["negative","neutral","positive"])
    plt.title("Ratings vs Rule-Based Sentiment")
    plt.savefig(os.path.join(CHARTS_DIR, "ratings_vs_sentiment.png"), bbox_inches="tight")
    plt.close()

    # 3. Topic cluster distribution (counts per topic)
    plt.figure(figsize=(8,4))
    sns.countplot(x="topic", data=topics_df)
    plt.title("Topic cluster distribution")
    plt.savefig(os.path.join(CHARTS_DIR, "topic_clusters.png"), bbox_inches="tight")
    plt.close()

    # 4. Wordclouds for positive and negative
    for label, fname in [("positive","wordcloud_positive.png"), ("negative","wordcloud_negative.png")]:
        texts = " ".join(top for top in df[df["rule_sentiment"]==label]["cleaned_text"])
        if len(texts.strip())==0: texts = "no data"
        wc = WordCloud(width=800, height=400, background_color="white").generate(texts)
        wc.to_file(os.path.join(CHARTS_DIR, fname))

    print("Charts saved to", CHARTS_DIR)

# -----------------------------
# Part 7: Report Generation (PDF)
# -----------------------------
def generate_pdf_report(data_summary_path, sentiment_metrics_path, topics_path, insights_path, charts_dir, output_pdf):
    doc = SimpleDocTemplate(output_pdf, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("Customer Reviews Analysis Report", styles["Title"]))
    story.append(Spacer(1,12))

    # Data summary
    with open(data_summary_path, "r", encoding="utf-8") as f:
        data_summary = json.load(f)
    story.append(Paragraph("Data Summary:", styles["Heading2"]))
    for k,v in data_summary.items():
        story.append(Paragraph(f"{k}: {v}", styles["Normal"]))
    story.append(Spacer(1,12))

    # Sentiment metrics
    with open(sentiment_metrics_path, "r", encoding="utf-8") as f:
        sentiment_metrics = json.load(f)
    story.append(Paragraph("Sentiment Model Metrics:", styles["Heading2"]))
    story.append(Paragraph(f"Accuracy: {sentiment_metrics.get('accuracy', None)}", styles["Normal"]))
    story.append(Spacer(1,12))

    # Topics
    with open(topics_path, "r", encoding="utf-8") as f:
        topics = json.load(f)
    story.append(Paragraph("Extracted Topics (top keywords):", styles["Heading2"]))
    for t, kw in topics.items():
        story.append(Paragraph(f"{t}: {', '.join(kw[:8])}", styles["Normal"]))
    story.append(Spacer(1,12))

    # Insert charts images
    story.append(Paragraph("Charts:", styles["Heading2"]))
    for fname in ["sentiment_distribution.png","ratings_vs_sentiment.png","topic_clusters.png","wordcloud_positive.png","wordcloud_negative.png"]:
        p = os.path.join(charts_dir, fname)
        if os.path.exists(p):
            story.append(Image(p, width=450, height=250))
            story.append(Spacer(1,12))

    # Insights summary
    with open(insights_path, "r", encoding="utf-8") as f:
        insights = json.load(f)
    story.append(Paragraph("Key Insights:", styles["Heading2"]))
    story.append(Paragraph("Top positive keywords: " + ", ".join(insights.get("top_positive_keywords", [])), styles["Normal"]))
    story.append(Paragraph("Top negative keywords: " + ", ".join(insights.get("top_negative_keywords", [])), styles["Normal"]))
    story.append(Spacer(1,12))

    doc.build(story)
    print(f"PDF report generated: {output_pdf}")

# -----------------------------
# Orchestration main()
# -----------------------------
def main():
    print("Starting end-to-end Customer Review Intelligence pipeline...")
    df = load_and_clean_csv(DATA_PATH)

    # Text cleaning
    print("Cleaning text ...")
    df["cleaned_text"] = df["review_text"].apply(clean_text)

    # Rule-based sentiment
    print("Applying rule-based sentiment ...")
    df["rule_sentiment"] = df["cleaned_text"].apply(rule_based_sentiment)

    # Train ML classifier for sentiment
    print("Training ML sentiment classifier ...")
    vectorizer, classifier, metrics = train_sentiment_classifier(df)

    # Topic extraction
    print("Extracting topics (KMeans) ...")
    topics_df, topics, kmeans_model, topic_vectorizer = extract_topics(df, n_clusters=6)

    # Insights
    print("Generating insights ...")
    insights = generate_insights(df, topics)

    # Charts
    print("Creating charts ...")
    make_charts(df, topics_df)

    # Save final outputs (already saved some)
    save_json(metrics, os.path.join(OUTPUT_DIR, "sentiment_metrics.json"))
    save_json(topics, os.path.join(OUTPUT_DIR, "topics.json"))
    save_json(insights, os.path.join(OUTPUT_DIR, "insights.json"))

    # Generate PDF report
    report_pdf = os.path.join(OUTPUT_DIR, "final_review_analysis_report.pdf")
    generate_pdf_report(
        data_summary_path=os.path.join(OUTPUT_DIR, "data_summary.json"),
        sentiment_metrics_path=os.path.join(OUTPUT_DIR, "sentiment_metrics.json"),
        topics_path=os.path.join(OUTPUT_DIR, "topics.json"),
        insights_path=os.path.join(OUTPUT_DIR, "insights.json"),
        charts_dir=CHARTS_DIR,
        output_pdf=report_pdf
    )

    # Save small CSV with topics and sentiments for inspection
    topics_df.to_csv(os.path.join(OUTPUT_DIR, "reviews_with_topics.csv"), index=False)
    print("Pipeline finished. Outputs under", OUTPUT_DIR)

if __name__ == "__main__":
    main()
