"""
SEO Content Quality & Duplicate Detector - Streamlit App
BONUS: Deploy this to Streamlit Cloud for extra points
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from utils.parser import parse_html_content, scrape_url
from utils.features import extract_features
from utils.scorer import load_model, predict_quality, find_similar_content

# Page config
st.set_page_config(
    page_title="SEO Content Analyzer",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .quality-high {
        color: #28a745;
        font-weight: bold;
    }
    .quality-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .quality-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load model and data
@st.cache_resource
def load_resources():
    """Load model and historical data."""
    model_data = load_model()
    
    # Load historical data for duplicate detection
    try:
        df_features = pd.read_csv('../data/features.csv')
        return model_data, df_features
    except:
        return model_data, None

model_data, df_historical = load_resources()

# Header
st.markdown('<h1 class="main-header">🔍 SEO Content Quality Analyzer</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📋 About")
    st.write("""
    This tool analyzes web content for:
    - **Quality scoring** (Low/Medium/High)
    - **Readability metrics**
    - **Duplicate detection**
    - **Thin content identification**
    """)
    
    st.header("⚙️ Settings")
    similarity_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.5,
        max_value=0.95,
        value=0.80,
        step=0.05,
        help="Threshold for duplicate detection"
    )
    
    st.markdown("---")
    st.markdown("**Dataset Statistics**")
    if df_historical is not None:
        st.metric("Total Pages", len(df_historical))
        st.metric("Avg Word Count", f"{df_historical['word_count'].mean():.0f}")

# Main content
tab1, tab2 = st.tabs(["🔍 Analyze URL", "📊 Batch Analysis"])

# Tab 1: Single URL Analysis
with tab1:
    st.subheader("Analyze a Single URL")
    
    url_input = st.text_input(
        "Enter URL to analyze:",
        placeholder="https://example.com/article"
    )
    
    analyze_button = st.button("🚀 Analyze", type="primary")
    
    if analyze_button and url_input:
        with st.spinner("Analyzing content..."):
            try:
                # Scrape and parse
                html_content = scrape_url(url_input)
                
                if not html_content:
                    st.error("❌ Failed to fetch content from URL")
                else:
                    parsed = parse_html_content(html_content)
                    
                    if parsed['word_count'] == 0:
                        st.error("❌ No content extracted from page")
                    else:
                        # Extract features
                        features = extract_features(parsed['body_text'])
                        
                        # Predict quality
                        prediction = predict_quality(model_data, features)
                        
                        # Find similar content
                        similar = []
                        if df_historical is not None:
                            similar = find_similar_content(
                                features['embedding'],
                                df_historical,
                                similarity_threshold
                            )
                        
                        # Display results
                        st.success("✅ Analysis complete!")
                        
                        # Metrics row
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Word Count", features['word_count'])
                        
                        with col2:
                            st.metric("Sentences", features['sentence_count'])
                        
                        with col3:
                            st.metric("Readability Score", f"{features['flesch_reading_ease']:.1f}")
                        
                        with col4:
                            quality_class = f"quality-{prediction['quality_label'].lower()}"
                            st.markdown(
                                f'<div class="metric-card"><p>Quality</p><p class="{quality_class}">{prediction["quality_label"]}</p></div>',
                                unsafe_allow_html=True
                            )
                        
                        # Title
                        st.markdown("### 📄 Page Information")
                        st.write(f"**Title:** {parsed['title']}")
                        st.write(f"**URL:** {url_input}")
                        
                        # Quality details
                        st.markdown("### 🎯 Quality Assessment")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Confidence Scores:**")
                            for label, score in prediction['confidence'].items():
                                st.progress(score, text=f"{label}: {score:.1%}")
                        
                        with col2:
                            st.write("**Content Flags:**")
                            is_thin = features['word_count'] < 500
                            st.write(f"- Thin Content: {'⚠️ Yes' if is_thin else '✅ No'}")
                            st.write(f"- Word Count: {'✅ Good' if features['word_count'] > 1000 else '⚠️ Low'}")
                            st.write(f"- Readability: {'✅ Good' if 30 < features['flesch_reading_ease'] < 70 else '⚠️ Check'}")
                        
                        # Keywords
                        if features.get('top_keywords'):
                            st.markdown("### 🔑 Top Keywords")
                            keywords = features['top_keywords'].split('|')
                            st.write(", ".join([f"`{kw}`" for kw in keywords]))
                        
                        # Similar content
                        if similar:
                            st.markdown("### 🔄 Similar Content Detected")
                            st.warning(f"Found {len(similar)} similar page(s)")
                            
                            for item in similar[:5]:
                                with st.expander(f"Similarity: {item['similarity']:.1%}"):
                                    st.write(f"**URL:** {item['url']}")
                        else:
                            st.markdown("### ✅ No Duplicates Detected")
                            st.info("This content appears to be unique.")
                        
                        # Content preview
                        with st.expander("📝 Content Preview"):
                            preview_text = parsed['body_text'][:500]
                            st.text(preview_text + "...")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Tab 2: Batch Analysis
with tab2:
    st.subheader("Batch URL Analysis")
    
    st.write("Upload a CSV file with a 'url' column to analyze multiple URLs.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            
            if 'url' not in df_batch.columns:
                st.error("❌ CSV must contain a 'url' column")
            else:
                st.write(f"Found {len(df_batch)} URLs")
                st.dataframe(df_batch.head())
                
                if st.button("🚀 Analyze Batch", type="primary"):
                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, row in df_batch.iterrows():
                        url = row['url']
                        status_text.text(f"Processing {idx+1}/{len(df_batch)}: {url[:50]}...")
                        
                        try:
                            html_content = scrape_url(url)
                            if html_content:
                                parsed = parse_html_content(html_content)
                                features = extract_features(parsed['body_text'])
                                prediction = predict_quality(model_data, features)
                                
                                results.append({
                                    'url': url,
                                    'word_count': features['word_count'],
                                    'readability': features['flesch_reading_ease'],
                                    'quality': prediction['quality_label'],
                                    'is_thin': features['word_count'] < 500
                                })
                            else:
                                results.append({
                                    'url': url,
                                    'word_count': 0,
                                    'readability': 0,
                                    'quality': 'Error',
                                    'is_thin': True
                                })
                        except:
                            results.append({
                                'url': url,
                                'word_count': 0,
                                'readability': 0,
                                'quality': 'Error',
                                'is_thin': True
                            })
                        
                        progress_bar.progress((idx + 1) / len(df_batch))
                    
                    status_text.text("✅ Analysis complete!")
                    
                    # Show results
                    df_results = pd.DataFrame(results)
                    st.dataframe(df_results)
                    
                    # Summary
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Analyzed", len(df_results))
                    
                    with col2:
                        high_quality = (df_results['quality'] == 'High').sum()
                        st.metric("High Quality", high_quality)
                    
                    with col3:
                        thin_content = df_results['is_thin'].sum()
                        st.metric("Thin Content", thin_content)
                    
                    # Download results
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="seo_analysis_results.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Built with Streamlit | SEO Content Analyzer v1.0</p>
</div>
""", unsafe_allow_html=True)