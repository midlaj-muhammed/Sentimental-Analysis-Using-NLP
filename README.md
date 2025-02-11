# Multi-Model Sentiment Analysis System 🎭

A robust and comprehensive sentiment analysis system that leverages multiple state-of-the-art models to provide accurate sentiment predictions for text input.

## 🌟 Features

- **Ensemble Approach**: Combines multiple sentiment analysis models:
  - BERT (Multi-lingual)
  - Random Forest Classifier
  - TextBlob
  - VADER Sentiment
- **Web Interface**: User-friendly Flask web application
- **Multi-lingual Support**: Can analyze text in multiple languages
- **High Accuracy**: Ensemble approach provides more reliable sentiment predictions
- **Real-time Analysis**: Instant sentiment analysis results

## 🛠️ Technologies Used

- Python 3.x
- Flask
- PyTorch
- Transformers (Hugging Face)
- scikit-learn
- TextBlob
- VADER Sentiment
- pandas
- NumPy

## 📦 Installation

1. Clone the repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the pre-trained models (will be downloaded automatically on first run)

## 🚀 Usage

1. Start the Flask application:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to `http://localhost:5000`
3. Enter the text you want to analyze
4. Get comprehensive sentiment analysis results!

## 🏗️ Project Structure

- `app.py`: Main Flask application and API endpoints
- `sentimentalanalysisnlp.py`: Core sentiment analysis implementation
- `templates/`: HTML templates for the web interface
- `requirements.txt`: Project dependencies
- `sentiment_model.joblib`: Trained Random Forest model
- `vectorizer.joblib`: TF-IDF vectorizer
- `dataset.csv`: Training dataset

## 🎯 Model Details

The system uses an ensemble approach combining:
- **BERT**: Multi-lingual BERT model for deep learning-based sentiment analysis
- **Random Forest**: ML-based classification using TF-IDF features
- **TextBlob**: Rule-based sentiment analysis
- **VADER**: Specifically attuned to sentiments expressed in social media

## 📊 Performance

The system uses multiple models to provide more robust and accurate sentiment predictions, with each model contributing its strengths:
- BERT: Excellent at understanding context and nuanced expressions
- Random Forest: Good at domain-specific patterns
- TextBlob & VADER: Strong at handling informal text and social media content

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check issues page.

## 📝 License

This project is open source and available under the MIT License.

---
Created with ❤️ for IBM Project
