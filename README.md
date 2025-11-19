# Customer Review Intelligence System

## Overview
This project is an end-to-end AI pipeline for analyzing customer reviews. It ingests review data, cleans and preprocesses text, performs sentiment analysis, extracts key topics, identifies top complaints & praises, trains a sentiment classifier, generates visualizations, and exports a final report.  

This simulates a production-grade workflow commonly used in business intelligence and customer experience analytics.

---

## Project Structure
project/
│
├─ data/
│ └─ customer_reviews.csv # Raw review data
│
├─ models/
│ └─ sentiment_model.pkl # Trained sentiment classifier
│
├─ output/
│ ├─ data_summary.json # Basic dataset stats
│ ├─ insights.json # Top keywords & sentiment insights
│ ├─ sentiment_metrics.json # ML classifier metrics
│ ├─ topics.json # Topic modeling results
│ ├─ charts/
│ │ ├─ sentiment_distribution.png
│ │ ├─ wordcloud_positive.png
│ │ ├─ wordcloud_negative.png
│ │ └─ topic_clusters.png
│ └─ final_review_analysis_report.pdf # Comprehensive report
│
└─ main.py # Main pipeline script


---

## Features

1. **Data Ingestion**
   - Loads CSV data with `pandas`.
   - Handles UTF-8 and Latin-1 encoding.
   - Drops empty reviews and validates required columns.
   
2. **Text Preprocessing**
   - Lowercasing, punctuation & number removal.
   - Tokenization, stopword removal.
   - Lemmatization.
   
3. **Sentiment Analysis**
   - **Rule-Based:** Using VADER sentiment scores.
   - **Machine Learning:** Logistic Regression trained on TF-IDF features.
   - Metrics saved in `sentiment_metrics.json`.
   
4. **Topic Extraction**
   - TF-IDF vectorization + KMeans clustering.
   - Extracts top 10 keywords per topic.
   - Visualizes cluster distribution (`topic_clusters.png`).
   
5. **Insights Generation**
   - Top positive & negative keywords.
   - Sentiment distribution counts.
   - Average rating.
   - Saved in `insights.json`.
   
6. **Visualizations**
   - Sentiment distribution bar chart.
   - Word clouds for positive & negative reviews.
   - Topic cluster visualization.
   
7. **Report Generation**
   - Comprehensive PDF report including:
     - Dataset overview
     - Sentiment metrics
     - Top keywords & topics
     - Charts & graphs
   - Saved as `final_review_analysis_report.pdf`.

---

## Requirements

**Python Libraries:**
- pandas
- numpy
- nltk
- scikit-learn
- matplotlib
- seaborn
- wordcloud
- fpdf

Install dependencies with:

```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn wordcloud fpdf
