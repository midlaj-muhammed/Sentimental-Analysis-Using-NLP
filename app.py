from flask import Flask, render_template, request
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import pandas as pd
from scipy.special import softmax
import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class SentimentAnalyzer:
    def __init__(self):
        try:
            # Initialize models and analyzers
            self.MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
            self.vader = SentimentIntensityAnalyzer()
            
            # Initialize ML components
            self.vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8)
            self.classifier = RandomForestClassifier(n_estimators=200, random_state=0)
            
            # Load and train ML model if not already trained
            self.train_ml_model()

            # Move BERT model to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            logger.info(f"Models loaded successfully. Using device: {self.device}")
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    def train_ml_model(self):
        """Train ML model using dataset.csv"""
        model_path = 'sentiment_model.joblib'
        vectorizer_path = 'vectorizer.joblib'
        
        # Check if trained model exists
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            logger.info("Loading pre-trained model and vectorizer...")
            self.classifier = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            return

        try:
            logger.info("Loading and preprocessing dataset...")
            # Load dataset
            dataset = pd.read_csv('dataset.csv')
            
            # Preprocess text data
            texts = dataset['text'].fillna('')
            processed_texts = [self.clean_text(text) for text in texts]
            
            # Convert sentiment labels
            sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
            labels = dataset['airline_sentiment'].map(sentiment_map)
            
            # Split dataset
            X_train, X_test, y_train, y_test = train_test_split(
                processed_texts, labels, test_size=0.2, random_state=42
            )
            
            # Vectorize text
            logger.info("Vectorizing text data...")
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
            
            # Train model
            logger.info("Training Random Forest classifier...")
            self.classifier.fit(X_train_vec, y_train)
            
            # Evaluate model
            y_pred = self.classifier.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Model accuracy: {accuracy:.4f}")
            
            # Save model and vectorizer
            joblib.dump(self.classifier, model_path)
            joblib.dump(self.vectorizer, vectorizer_path)
            logger.info("Model and vectorizer saved successfully")
            
        except Exception as e:
            logger.error(f"Error in train_ml_model: {str(e)}")
            raise

    def clean_text(self, text):
        """Clean and preprocess text"""
        try:
            # Convert to lowercase
            text = str(text).lower()
            
            # Remove URLs
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            
            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            
            # Remove special characters but keep emojis
            text = re.sub(r'[^\w\s!?.,\'\"😊😔😐👍❤️💕😍]', ' ', text)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            return text
        except Exception as e:
            logger.error(f"Error in clean_text: {str(e)}")
            return text

    def get_bert_sentiment(self, text):
        """Get sentiment scores from BERT model"""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
                scores = scores.cpu().numpy()[0]
            
            # Convert 1-5 scale to -1 to 1 scale
            bert_score = (np.argmax(scores) + 1 - 3) / 2
            return bert_score, scores
        except Exception as e:
            logger.error(f"Error in get_bert_sentiment: {str(e)}")
            return 0.0, np.zeros(5)

    def get_ml_sentiment(self, text):
        """Get sentiment from trained ML model"""
        try:
            # Preprocess and vectorize text
            cleaned_text = self.clean_text(text)
            text_vec = self.vectorizer.transform([cleaned_text])
            
            # Get prediction
            prediction = self.classifier.predict(text_vec)[0]
            
            # Get probability scores
            proba = self.classifier.predict_proba(text_vec)[0]
            
            return prediction, proba
        except Exception as e:
            logger.error(f"Error in get_ml_sentiment: {str(e)}")
            return 0, np.array([0.33, 0.34, 0.33])

    def get_textblob_sentiment(self, text):
        """Get sentiment from TextBlob"""
        try:
            analysis = TextBlob(text)
            return analysis.sentiment.polarity
        except Exception as e:
            logger.error(f"Error in get_textblob_sentiment: {str(e)}")
            return 0.0

    def get_vader_sentiment(self, text):
        """Get sentiment from VADER"""
        try:
            scores = self.vader.polarity_scores(text)
            return scores['compound']
        except Exception as e:
            logger.error(f"Error in get_vader_sentiment: {str(e)}")
            return 0.0

    def analyze_sentiment(self, text):
        """
        Ensemble sentiment analysis using BERT, ML model, TextBlob, and VADER
        """
        try:
            # Clean the text
            cleaned_text = self.clean_text(text)
            
            # Get sentiments from different models
            bert_score, bert_raw_scores = self.get_bert_sentiment(cleaned_text)
            ml_score, ml_proba = self.get_ml_sentiment(cleaned_text)
            textblob_score = self.get_textblob_sentiment(cleaned_text)
            vader_score = self.get_vader_sentiment(cleaned_text)
            
            # Weighted ensemble (adjust weights based on model performance)
            weights = {
                'bert': 0.3,
                'ml': 0.3,
                'textblob': 0.2,
                'vader': 0.2
            }
            
            # Calculate weighted average
            ensemble_score = (
                bert_score * weights['bert'] +
                ml_score * weights['ml'] +
                textblob_score * weights['textblob'] +
                vader_score * weights['vader']
            )
            
            # Additional features
            features = {
                'text_length': len(cleaned_text.split()),
                'exclamation_marks': cleaned_text.count('!'),
                'question_marks': cleaned_text.count('?'),
                'has_emojis': bool(re.search(r'[😊😔😐👍❤️💕😍]', cleaned_text))
            }
            
            # Adjust score based on features
            if features['has_emojis'] and ensemble_score > 0:
                ensemble_score *= 1.2
            if features['exclamation_marks'] > 0 and ensemble_score > 0:
                ensemble_score *= 1.1
            
            # Ensure score is between -1 and 1
            ensemble_score = max(min(ensemble_score, 1.0), -1.0)
            
            # Calculate confidence scores
            confidence_scores = {
                'negative': float(ml_proba[0]),
                'neutral': float(ml_proba[1]),
                'positive': float(ml_proba[2])
            }
            
            return {
                'compound': round(ensemble_score, 3),
                'bert_score': round(bert_score, 3),
                'ml_score': round(float(ml_score), 3),
                'textblob_score': round(textblob_score, 3),
                'vader_score': round(vader_score, 3),
                'confidence_scores': {
                    k: round(v * 100, 1)
                    for k, v in confidence_scores.items()
                },
                'sentiment': self.get_sentiment_label(ensemble_score)
            }
        except Exception as e:
            logger.error(f"Error in analyze_sentiment: {str(e)}")
            return {
                'compound': 0.0,
                'bert_score': 0.0,
                'ml_score': 0.0,
                'textblob_score': 0.0,
                'vader_score': 0.0,
                'confidence_scores': {'negative': 33.3, 'neutral': 33.4, 'positive': 33.3},
                'sentiment': ('Neutral', '😐')
            }

    def get_sentiment_label(self, score):
        """Convert score to sentiment label with dynamic thresholds"""
        try:
            if score >= 0.2:
                return "Positive", "😊"
            elif score <= -0.2:
                return "Negative", "😔"
            else:
                return "Neutral", "😐"
        except Exception as e:
            logger.error(f"Error in get_sentiment_label: {str(e)}")
            return "Neutral", "😐"

# Initialize sentiment analyzer
analyzer = SentimentAnalyzer()

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        if request.method == 'POST':
            text = request.form.get('text', '').strip()
            
            if not text:
                return render_template('index.html', 
                                    error="Please enter some text to analyze.")
            
            # Get sentiment analysis
            sentiment_scores = analyzer.analyze_sentiment(text)
            
            # Get sentiment label and emoji
            sentiment, emoji = sentiment_scores['sentiment']
            
            # Prepare scores for display
            scores = {
                'compound': sentiment_scores['compound'],
                'confidence': sentiment_scores['confidence_scores']
            }
            
            return render_template('index.html',
                                text=text,
                                sentiment=sentiment,
                                emoji=emoji,
                                scores=scores)
        
        return render_template('index.html')
    
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        return render_template('index.html',
                             error="An error occurred while analyzing the text.")

if __name__ == '__main__':
    app.run(debug=True)
