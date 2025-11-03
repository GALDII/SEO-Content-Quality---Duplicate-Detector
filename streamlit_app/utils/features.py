"""
Feature extraction utilities for SEO content analysis.
"""

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass


def clean_text(text):
    """Clean and normalize text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = " ".join(text.split())
    return text


def calculate_sentence_count(text):
    """Calculate number of sentences in text."""
    try:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
        return len(sentences)
    except:
        return text.count('.') + text.count('!') + text.count('?')


def calculate_readability(text):
    """Calculate Flesch Reading Ease score."""
    try:
        score = textstat.flesch_reading_ease(text)
        return round(score, 2)
    except:
        return 50.0


def calculate_avg_word_length(text):
    """Calculate average word length."""
    try:
        words = word_tokenize(text)
        words = [w for w in words if w.isalpha()]
        if len(words) == 0:
            return 0
        avg_length = sum(len(word) for word in words) / len(words)
        return round(avg_length, 2)
    except:
        words = text.split()
        words = [w for w in words if w.isalpha()]
        if len(words) == 0:
            return 0
        return round(sum(len(word) for word in words) / len(words), 2)


def extract_top_keywords(text, top_n=5):
    """Extract top keywords using TF-IDF."""
    try:
        stop_words = list(stopwords.words('english'))
        
        vectorizer = TfidfVectorizer(
            max_features=50,
            stop_words=stop_words,
            ngram_range=(1, 2),
            min_df=1
        )
        
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        
        doc_vector = tfidf_matrix.toarray().flatten()
        top_indices = doc_vector.argsort()[-top_n:][::-1]
        top_keywords = [feature_names[i] for i in top_indices if doc_vector[i] > 0]
        
        return "|".join(top_keywords)
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return ""


def generate_embedding_tfidf(text, max_features=100):
    """
    Generate TF-IDF based embedding as fallback.
    Returns a dense vector representation.
    """
    try:
        stop_words = list(stopwords.words('english'))
        
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=stop_words,
            ngram_range=(1, 2)
        )
        
        # Fit and transform
        tfidf_matrix = vectorizer.fit_transform([text])
        embedding = tfidf_matrix.toarray()[0]
        
        # Pad to consistent size if needed
        if len(embedding) < max_features:
            embedding = np.pad(embedding, (0, max_features - len(embedding)))
        
        return embedding
    except Exception as e:
        print(f"Error generating TF-IDF embedding: {e}")
        return np.zeros(max_features)

# This is the function you added
def calculate_unique_word_ratio(text):
    """Calculate the ratio of unique words to total words."""
    try:
        words = word_tokenize(text)
        words = [w for w in words if w.isalpha()]
        if len(words) == 0:
            return 0
        unique_words = set(words)
        return round(len(unique_words) / len(words), 4)
    except Exception:
        # Fallback for any errors
        words_list = text.split()
        if not words_list:
            return 0
        return round(len(set(words_list)) / len(words_list), 4)


def generate_embedding_transformer(text, model=None):
    """
    Generate embedding using SentenceTransformer.
    Falls back to TF-IDF if model is not available.
    """
    if model is not None:
        try:
            embedding = model.encode([text])[0]
            return embedding
        except Exception as e:
            print(f"Error with SentenceTransformer: {e}")
            return generate_embedding_tfidf(text)
    else:
        return generate_embedding_tfidf(text)


def extract_features(text, embedding_model=None):
    """
    Extract all features from text.
    
    Args:
        text: Input text to analyze
        embedding_model: Optional SentenceTransformer model
        
    Returns:
        dict: Dictionary of extracted features
    """
    try:
        # Clean text
        clean_text_val = clean_text(text)
        
        # Calculate basic features
        word_count = len(text.split())
        sentence_count = calculate_sentence_count(text)
        flesch_score = calculate_readability(text)
        avg_word_length = calculate_avg_word_length(text)
        
        # *** FIX 1: Call your new function ***
        unique_ratio = calculate_unique_word_ratio(clean_text_val)
        
        # Extract keywords
        keywords = extract_top_keywords(clean_text_val)
        
        # Generate embedding
        embedding = generate_embedding_transformer(clean_text_val, embedding_model)
        
        return {
            'word_count': word_count,
            'sentence_count': max(1, sentence_count),
            'flesch_reading_ease': flesch_score,
            'avg_word_length': avg_word_length,
            
            # *** FIX 2: Add the key to the dictionary ***
            'unique_word_ratio': unique_ratio,
            
            'top_keywords': keywords,
            'embedding': embedding,
            'clean_text': clean_text_val
        }
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return {
            'word_count': 0,
            'sentence_count': 1,
            'flesch_reading_ease': 50.0,
            'avg_word_length': 5.0,
            
            'unique_word_ratio': 0.4,
            
            'top_keywords': "",
            'embedding': np.zeros(100),
            'clean_text': ""
        }