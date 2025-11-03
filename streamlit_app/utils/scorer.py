"""
Content quality scoring and duplicate detection utilities - FIXED VERSION
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Fix path resolution
def get_model_path():
    """Get the correct path to the model file."""
    # Try multiple possible locations
    possible_paths = [
        Path(__file__).parent.parent / 'models' / 'quality_model.pkl',  # From utils folder
        Path('models') / 'quality_model.pkl',  # From project root
        Path('../models/quality_model.pkl'),  # Relative path
        Path('streamlit_app/models/quality_model.pkl'),  # Alternative structure
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    # If no model found, return first path for error message
    return possible_paths[0]

def load_model():
    """
    Load the trained quality model.
    Returns dict with model, vectorizer, and scaler.
    """
    try:
        model_path = get_model_path()
        print(f"Attempting to load model from: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        print(f"✅ Model loaded successfully from {model_path}")
        return model_data
    
    except FileNotFoundError as e:
        print(f"❌ Error loading model: {e}")
        print(f"Current working directory: {Path.cwd()}")
        print(f"Looking for model at: {model_path}")
        print("\nPlease ensure the model file exists. You may need to:")
        print("1. Run the training notebook first: notebooks/seo_pipeline.ipynb")
        print("2. Check that models/quality_model.pkl was created")
        return None
    
    except Exception as e:
        print(f"❌ Unexpected error loading model: {e}")
        return None

def predict_quality(model_data, features):
    """
    Predict content quality for given features.
    
    Args:
        model_data: Dict with model, vectorizer, scaler
        features: Dict of extracted features
    
    Returns:
        Dict with quality_label and confidence scores
    """
    if model_data is None:
        # Fallback prediction based on simple rules
        return fallback_quality_prediction(features)
    
    try:
        # Prepare feature vector
        feature_vector = np.array([[
            features['word_count'],
            features['sentence_count'],
            features['avg_word_length'],
            features['flesch_reading_ease'],
            features['unique_word_ratio']
        ]])
        
        # Scale features
        feature_vector_scaled = model_data['scaler'].transform(feature_vector)
        
        # Predict
        prediction = model_data['model'].predict(feature_vector_scaled)[0]
        probabilities = model_data['model'].predict_proba(feature_vector_scaled)[0]
        
        # Get confidence scores for each class
        classes = model_data['model'].classes_
        confidence = dict(zip(classes, probabilities))
        
        return {
            'quality_label': prediction,
            'confidence': confidence
        }
    
    except Exception as e:
        print(f"⚠️ Error in prediction: {e}")
        return fallback_quality_prediction(features)

def fallback_quality_prediction(features):
    """
    Simple rule-based quality prediction when model is unavailable.
    """
    word_count = features['word_count']
    readability = features['flesch_reading_ease']
    
    # Simple scoring logic
    score = 0
    
    # Word count scoring
    if word_count >= 1000:
        score += 3
    elif word_count >= 500:
        score += 2
    elif word_count >= 300:
        score += 1
    
    # Readability scoring
    if 30 <= readability <= 70:
        score += 2
    elif 20 <= readability <= 80:
        score += 1
    
    # Unique word ratio
    if features['unique_word_ratio'] > 0.5:
        score += 1
    
    # Determine quality
    if score >= 5:
        quality = 'High'
        confidence = {'High': 0.7, 'Medium': 0.2, 'Low': 0.1}
    elif score >= 3:
        quality = 'Medium'
        confidence = {'High': 0.2, 'Medium': 0.6, 'Low': 0.2}
    else:
        quality = 'Low'
        confidence = {'High': 0.1, 'Medium': 0.2, 'Low': 0.7}
    
    return {
        'quality_label': quality,
        'confidence': confidence
    }

def find_similar_content(embedding, df_historical, threshold=0.8):
    """
    Find similar content in historical data using embeddings or TF-IDF.
    
    Args:
        embedding: Content embedding vector or None for TF-IDF fallback
        df_historical: DataFrame with historical content
        threshold: Similarity threshold (0-1)
    
    Returns:
        List of dicts with similar content URLs and similarity scores
    """
    if df_historical is None or len(df_historical) == 0:
        return []
    
    try:
        # Check if embeddings column exists
        if 'embedding' not in df_historical.columns:
            print("⚠️ No embeddings found in historical data")
            return []
        
        if embedding is None:
            print("⚠️ No embedding provided for comparison")
            return []
        
        # Convert embedding to numpy array if needed
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        
        # Ensure 2D array
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        # Get historical embeddings
        historical_embeddings = np.vstack(df_historical['embedding'].values)
        
        # Calculate similarities
        similarities = cosine_similarity(embedding, historical_embeddings)[0]
        
        # Find items above threshold
        similar_indices = np.where(similarities >= threshold)[0]
        
        # Create results
        similar_content = []
        for idx in similar_indices:
            similar_content.append({
                'url': df_historical.iloc[idx]['url'],
                'similarity': float(similarities[idx])
            })
        
        # Sort by similarity (descending)
        similar_content.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similar_content
    
    except Exception as e:
        print(f"⚠️ Error in similarity detection: {e}")
        return []

def get_quality_insights(features, prediction):
    """
    Generate actionable insights based on content analysis.
    
    Args:
        features: Dict of extracted features
        prediction: Dict with quality prediction
    
    Returns:
        List of insight strings
    """
    insights = []
    
    # Word count insights
    word_count = features['word_count']
    if word_count < 300:
        insights.append("⚠️ Content is too short (thin content). Aim for at least 300 words.")
    elif word_count < 500:
        insights.append("📝 Consider expanding content to 500+ words for better SEO.")
    elif word_count > 2000:
        insights.append("✅ Excellent content length! Great for comprehensive coverage.")
    else:
        insights.append("✅ Good content length for SEO.")
    
    # Readability insights
    readability = features['flesch_reading_ease']
    if readability < 30:
        insights.append("⚠️ Content is difficult to read. Simplify language and sentence structure.")
    elif readability > 70:
        insights.append("📖 Content is very easy to read. Ensure it maintains authority.")
    else:
        insights.append("✅ Readability is well-balanced.")
    
    # Sentence structure
    avg_sentence_length = word_count / features['sentence_count'] if features['sentence_count'] > 0 else 0
    if avg_sentence_length > 25:
        insights.append("⚠️ Sentences are too long. Break them into shorter sentences.")
    elif avg_sentence_length < 10:
        insights.append("📝 Sentences are very short. Consider varying sentence length.")
    
    # Vocabulary diversity
    if features['unique_word_ratio'] < 0.3:
        insights.append("⚠️ Low vocabulary diversity. Use more varied words.")
    elif features['unique_word_ratio'] > 0.6:
        insights.append("✅ Excellent vocabulary diversity!")
    
    # Quality prediction insight
    quality = prediction['quality_label']
    confidence = max(prediction['confidence'].values())
    
    if quality == 'High':
        insights.append(f"✅ High-quality content detected! (Confidence: {confidence:.1%})")
    elif quality == 'Medium':
        insights.append(f"📊 Medium-quality content. Room for improvement. (Confidence: {confidence:.1%})")
    else:
        insights.append(f"⚠️ Low-quality content. Significant improvements needed. (Confidence: {confidence:.1%})")
    
    return insights

def calculate_content_score(features):
    """
    Calculate a simple 0-100 content score.
    
    Args:
        features: Dict of extracted features
    
    Returns:
        Float score between 0-100
    """
    score = 0
    
    # Word count (max 30 points)
    word_count = features['word_count']
    if word_count >= 1000:
        score += 30
    elif word_count >= 500:
        score += 20
    elif word_count >= 300:
        score += 10
    
    # Readability (max 30 points)
    readability = features['flesch_reading_ease']
    if 40 <= readability <= 60:
        score += 30
    elif 30 <= readability <= 70:
        score += 20
    elif 20 <= readability <= 80:
        score += 10
    
    # Vocabulary diversity (max 20 points)
    unique_ratio = features['unique_word_ratio']
    if unique_ratio >= 0.5:
        score += 20
    elif unique_ratio >= 0.4:
        score += 15
    elif unique_ratio >= 0.3:
        score += 10
    
    # Sentence structure (max 20 points)
    avg_sentence_length = word_count / features['sentence_count'] if features['sentence_count'] > 0 else 0
    if 15 <= avg_sentence_length <= 20:
        score += 20
    elif 10 <= avg_sentence_length <= 25:
        score += 15
    elif 8 <= avg_sentence_length <= 30:
        score += 10
    
    return min(score, 100)