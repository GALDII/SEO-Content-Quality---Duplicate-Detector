"""
Feature Extraction utilities for SEO content analysis
"""
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Global flag to track if sentence transformers is available
_SENTENCE_TRANSFORMERS_AVAILABLE = False
_embedding_model = None

def _try_load_sentence_transformer():
    """Try to load SentenceTransformer, return None if fails"""
    global _SENTENCE_TRANSFORMERS_AVAILABLE, _embedding_model
    try:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        _SENTENCE_TRANSFORMERS_AVAILABLE = True
        return _embedding_model
    except Exception as e:
        print(f"⚠️ Warning: Could not load SentenceTransformer: {e}")
        print("📝 Falling back to TF-IDF for similarity detection")
        _SENTENCE_TRANSFORMERS_AVAILABLE = False
        return None

def get_embedding_model():
    """Lazy load the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = _try_load_sentence_transformer()
    return _embedding_model

def clean_text(text):
    """
    Clean and normalize text.
    
    Args:
        text: Raw text string
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    # Lowercase and remove extra whitespace
    text = text.lower()
    text = " ".join(text.split())
    return text

def calculate_sentence_count(text):
    """
    Calculate number of sentences in text.
    
    Args:
        text: Text string
        
    Returns:
        int: Number of sentences
    """
    try:
        sentences = sent_tokenize(text)
        return len(sentences)
    except:
        return 0

def calculate_readability(text):
    """
    Calculate Flesch Reading Ease score.
    
    Args:
        text: Text string
        
    Returns:
        float: Flesch Reading Ease score (0-100)
    """
    try:
        score = textstat.flesch_reading_ease(text)
        return round(score, 2)
    except:
        return 0.0

def extract_keywords(text, top_n=5):
    """
    Extract top keywords using TF-IDF.
    
    Args:
        text: Text string
        top_n: Number of top keywords to extract
        
    Returns:
        str: Pipe-separated keywords
    """
    try:
        # Remove stop words
        stop_words = list(stopwords.words('english'))
        
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words=stop_words,
            ngram_range=(1, 2),
            min_df=1
        )
        
        # TF-IDF needs a list of documents
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        
        # Get top keywords for this document
        doc_vector = tfidf_matrix.toarray().flatten()
        top_indices = doc_vector.argsort()[-top_n:][::-1]
        top_keywords = [feature_names[i] for i in top_indices if doc_vector[i] > 0]
        
        return "|".join(top_keywords)
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return ""

def generate_embedding(text):
    """
    Generate sentence embedding using SentenceTransformers.
    Falls back to TF-IDF vector if SentenceTransformers unavailable.
    
    Args:
        text: Text string
        
    Returns:
        numpy array: Embedding vector
    """
    global _SENTENCE_TRANSFORMERS_AVAILABLE
    
    # Try sentence transformers first
    if _SENTENCE_TRANSFORMERS_AVAILABLE or _embedding_model is None:
        model = get_embedding_model()
        if model is not None:
            try:
                embedding = model.encode([text])[0]
                return embedding
            except Exception as e:
                print(f"Error with SentenceTransformer: {e}")
                _SENTENCE_TRANSFORMERS_AVAILABLE = False
    
    # Fallback: Use TF-IDF vector
    try:
        vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
        tfidf_vec = vectorizer.fit_transform([text]).toarray()[0]
        # Pad to 384 dimensions if needed
        if len(tfidf_vec) < 384:
            tfidf_vec = np.pad(tfidf_vec, (0, 384 - len(tfidf_vec)), 'constant')
        return tfidf_vec[:384]
    except:
        return np.zeros(384)

def extract_features(text):
    """
    Extract all features from text.
    
    Args:
        text: Raw text string
        
    Returns:
        dict: Dictionary containing all features
    """
    # Clean text
    clean_text_val = clean_text(text)
    
    # Calculate features
    word_count = len(text.split())
    sentence_count = calculate_sentence_count(text)
    readability = calculate_readability(text)
    keywords = extract_keywords(clean_text_val)
    embedding = generate_embedding(clean_text_val)
    
    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'flesch_reading_ease': readability,
        'top_keywords': keywords,
        'embedding': embedding,
        'clean_text': clean_text_val
    }