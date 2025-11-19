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
MODEL_DIR = "models"

# Ensure output folders exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------
# Utility Functions
# -------------------
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
    
    # Drop missing comments
    df = df.dropna(subset=['Comments'])
    print(f"Dataset loaded. Total reviews: {len(df)}")
    return df

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

# -------------------
# Sentiment Analysis
# -------------------
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
    print("\nML Classifier Performance:")
    print(classification_report(y_test, y_pred))
    
    metrics = {"accuracy": accuracy_score(y_test, y_pred)}
    
    # Save model and vectorizer
    import pickle
    with open(os.path.join(MODEL_DIR, "sentiment_model.pkl"), "wb") as f:
        pickle.dump({"model": clf, "vectorizer": vectorizer}, f)
    
    with open(os.path.join(OUTPUT_DIR, "sentiment_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    return clf, vectorizer, metrics

# -------------------
# Topic Extraction
# -------------------
def extract_topics(df, n_clusters=5):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(df['cleaned_text'])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['topic'] = kmeans.fit_predict(X_vec)
    
    terms = vectorizer.get_feature_names_out()
    topics = {}
    for i in range(n_clusters):
        idxs = np.argsort(kmeans.cluster_centers_[i])[::-1][:10]
        keywords = [terms[j] for j in idxs]
        topics[f"Topic_{i}"] = keywords
    
    with open(os.path.join(OUTPUT_DIR, "topics.json"), "w") as f:
        json.dump(topics, f, indent=4)
    
    # Topic cluster visualization
    plt.figure(figsize=(8,6))
    sns.countplot(df['topic'])
    plt.title("Topic Cluster Distribution")
    plt.xlabel("Topic")
    plt.ylabel("Number of Reviews")
    plt.savefig(os.path.join(CHARTS_DIR, "topic_clusters.png"))
    plt.close()
    
    return topics

# -------------------
# Visualization
# -------------------
def generate_charts(df):
    # Sentiment distribution
    plt.figure(figsize=(6,4))
    sns.countplot(df['sentiment_label'])
    plt.title("Sentiment Distribution")
    plt.savefig(os.path.join(CHARTS_DIR, "sentiment_distribution.png"))
    plt.close()
    
    # WordCloud Positive
    positive_text = " ".join(df[df['sentiment_label']=='Positive']['cleaned_text'])
    wc = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
    wc.to_file(os.path.join(CHARTS_DIR, "wordcloud_positive.png"))
    
    # WordCloud Negative
    negative_text = " ".join(df[df['sentiment_label']=='Negative']['cleaned_text'])
    wc = WordCloud(width=800, height=400, background_color='white').generate(negative_text)
    wc.to_file(os.path.join(CHARTS_DIR, "wordcloud_negative.png"))

# -------------------
# Insights Generation
# -------------------
def generate_insights(df):
    # Top positive/negative words
    pos_words = " ".join(df[df['sentiment_label']=='Positive']['cleaned_text']).split()
    neg_words = " ".join(df[df['sentiment_label']=='Negative']['cleaned_text']).split()
    from collections import Counter
    insights = {
        "top_positive_words": [w for w, _ in Counter(pos_words).most_common(5)],
        "top_negative_words": [w for w, _ in Counter(neg_words).most_common(5)],
        "sentiment_distribution": df['sentiment_label'].value_counts().to_dict(),
        "average_rating": df['Rating'].mean()
    }
    with open(os.path.join(OUTPUT_DIR, "insights.json"), "w") as f:
        json.dump(insights, f, indent=4)
    return insights

# -------------------
# PDF Report Generation
# -------------------
def generate_pdf_report(df, insights):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0,10,"Customer Review Analysis Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0,8,f"Total Reviews: {len(df)}")
    pdf.multi_cell(0,8,f"Average Rating: {df['Rating'].mean():.2f}")
    
    pdf.ln(5)
    pdf.multi_cell(0,8,"Top Positive Words: " + ", ".join(insights['top_positive_words']))
    pdf.multi_cell(0,8,"Top Negative Words: " + ", ".join(insights['top_negative_words']))
    
    # Add charts
    chart_files = [
        "sentiment_distribution.png",
        "wordcloud_positive.png",
        "wordcloud_negative.png",
        "topic_clusters.png"
    ]
    for chart in chart_files:
        chart_path = os.path.join(CHARTS_DIR, chart)
        if os.path.exists(chart_path):
            pdf.add_page()
            pdf.image(chart_path, x=15, y=30, w=180)
    
    pdf.output(os.path.join(OUTPUT_DIR, "final_review_analysis_report.pdf"))

# -------------------
# Main
# -------------------
def main():
    print("Starting Customer Review Intelligence pipeline...")
    
    df = load_data(DATA_PATH)
    
    df = add_rule_sentiment(df)
    
    clf, vectorizer, metrics = train_sentiment_classifier(df)
    
    topics = extract_topics(df)
    
    generate_charts(df)
    
    insights = generate_insights(df)
    
    generate_pdf_report(df, insights)
    
    print("Pipeline completed successfully!")
    print("ML Classifier Accuracy:", metrics['accuracy'])

if __name__ == "__main__":
    main()
