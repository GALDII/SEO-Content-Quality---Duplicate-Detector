import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

def load_model(model_path='../models/quality_model.pkl'):
    """
    Load the trained quality classification model.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        dict: Model data containing model, feature columns, etc.
    """
    try:
        model_path = Path(model_path)
        model_data = joblib.load(model_path)
        return model_data
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_quality(model_data, features):
    """
    Predict content quality using the trained model.
    
    Args:
        model_data: Dictionary containing the trained model
        features: Dictionary of extracted features
        
    Returns:
        dict: Prediction results (label and confidence scores)
    """
    try:
        if model_data is None:
            return {
                'quality_label': 'Unknown',
                'confidence': {'High': 0.33, 'Medium': 0.33, 'Low': 0.34}
            }
        
        model = model_data['model']
        feature_columns = model_data['feature_columns']
        
        # Prepare features array
        feature_values = np.array([[
            features['word_count'],
            features['sentence_count'],
            features['flesch_reading_ease']
        ]])
        
        # Predict
        quality_label = model.predict(feature_values)[0]
        quality_proba = model.predict_proba(feature_values)[0]
        
        # Get class names (usually ['High', 'Low', 'Medium'])
        class_names = model.classes_
        
        # Create confidence dictionary
        confidence = {class_names[i]: float(quality_proba[i]) 
                     for i in range(len(class_names))}
        
        return {
            'quality_label': quality_label,
            'confidence': confidence
        }
        
    except Exception as e:
        print(f"Error predicting quality: {e}")
        return {
            'quality_label': 'Error',
            'confidence': {'High': 0.0, 'Medium': 0.0, 'Low': 0.0}
        }

def find_similar_content(embedding, df_historical, threshold=0.80):
    """
    Find similar content in the historical dataset.
    
    Args:
        embedding: Embedding vector of the new content
        df_historical: DataFrame with historical embeddings
        threshold: Similarity threshold (0-1)
        
    Returns:
        list: List of similar content dictionaries
    """
    try:
        if df_historical is None or len(df_historical) == 0:
            return []
        
        # Convert historical embeddings from string to numpy array
        historical_embeddings = []
        for emb_str in df_historical['embedding']:
            try:
                emb = eval(emb_str) if isinstance(emb_str, str) else emb_str
                historical_embeddings.append(emb)
            except:
                continue
        
        if not historical_embeddings:
            return []
        
        historical_embeddings = np.array(historical_embeddings)
        
        # Compute similarities
        similarities = cosine_similarity([embedding], historical_embeddings)[0]
        
        # Find similar content above threshold
        similar_indices = np.where(similarities > threshold)[0]
        
        similar_content = []
        for idx in similar_indices:
            similar_content.append({
                'url': df_historical.iloc[idx]['url'],
                'similarity': float(similarities[idx])
            })
        
        # Sort by similarity (descending)
        similar_content = sorted(similar_content, 
                                key=lambda x: x['similarity'], 
                                reverse=True)
        
        return similar_content
        
    except Exception as e:
        print(f"Error finding similar content: {e}")
        return []

def calculate_quality_score(features):
    """
    Calculate a simple quality score based on features.
    This is a rule-based backup if model fails.
    
    Args:
        features: Dictionary of extracted features
        
    Returns:
        float: Quality score (0-100)
    """
    try:
        word_count = features['word_count']
        readability = features['flesch_reading_ease']
        
        # Word count score (0-40 points)
        if word_count > 2000:
            word_score = 40
        elif word_count > 1000:
            word_score = 30
        elif word_count > 500:
            word_score = 20
        else:
            word_score = 10
        
        # Readability score (0-40 points)
        if 30 <= readability <= 70:
            read_score = 40
        elif 20 <= readability <= 80:
            read_score = 30
        elif 10 <= readability <= 90:
            read_score = 20
        else:
            read_score = 10
        
        # Sentence structure (0-20 points)
        sentence_count = features['sentence_count']
        if sentence_count > 0:
            avg_words_per_sentence = word_count / sentence_count
            if 15 <= avg_words_per_sentence <= 25:
                structure_score = 20
            elif 10 <= avg_words_per_sentence <= 30:
                structure_score = 15
            else:
                structure_score = 10
        else:
            structure_score = 0
        
        total_score = word_score + read_score + structure_score
        
        return round(total_score, 1)
        
    except Exception as e:
        print(f"Error calculating quality score: {e}")
        return 0.0

def get_quality_insights(features, prediction):
    """
    Generate human-readable insights about content quality.
    
    Args:
        features: Dictionary of extracted features
        prediction: Prediction dictionary from predict_quality
        
    Returns:
        list: List of insight strings
    """
    insights = []
    
    word_count = features['word_count']
    readability = features['flesch_reading_ease']
    sentence_count = features['sentence_count']
    
    # Word count insights
    if word_count < 300:
        insights.append("⚠️ Very short content - consider expanding to at least 500 words")
    elif word_count < 500:
        insights.append("⚠️ Content is thin - aim for 500+ words for better SEO")
    elif word_count > 2000:
        insights.append("✅ Great content length!")
    else:
        insights.append("✅ Good content length")
    
    # Readability insights
    if readability < 30:
        insights.append("⚠️ Content is very difficult to read - simplify language")
    elif readability < 50:
        insights.append("ℹ️ Content is moderately difficult - consider simplifying")
    elif readability <= 70:
        insights.append("✅ Good readability level")
    else:
        insights.append("ℹ️ Very easy to read - ensure it's professional enough")
    
    # Sentence structure
    if sentence_count > 0:
        avg_words = word_count / sentence_count
        if avg_words > 30:
            insights.append("⚠️ Sentences are too long - break them up")
        elif avg_words < 10:
            insights.append("ℹ️ Sentences are very short - vary sentence length")
        else:
            insights.append("✅ Good sentence structure")
    
    # Quality prediction confidence
    confidence = prediction['confidence'][prediction['quality_label']]
    if confidence < 0.5:
        insights.append("ℹ️ Model has low confidence - content may be borderline")
    
    return insights