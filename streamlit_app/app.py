"""
SEO Content Quality & Duplicate Detector - Enhanced Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from utils.parser import parse_html_content, scrape_url
from utils.features import extract_features
from utils.scorer import load_model, predict_quality, find_similar_content, get_quality_insights

# Page config
st.set_page_config(
    page_title="SEO Content Analyzer Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    /* Main styles */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeIn 1s;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Quality badges */
    .quality-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        margin: 1rem 0;
    }
    
    .quality-high {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    .quality-medium {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .quality-low {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
    }
    
    /* Info cards */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Feature icons */
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    /* Tip card */
    .tip-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    
    .tip-title {
        font-size: 1.3rem;
        font-weight: bold;
        color: #667eea;
        margin-bottom: 1rem;
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

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

model_data, df_historical = load_resources()

# Header
st.markdown('<h1 class="main-header">🚀 SEO Content Analyzer Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Content Quality & Duplicate Detection Tool</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 About This Tool")
    
    st.info("""
    **SEO Content Analyzer Pro** uses machine learning to:
    
    ✨ Analyze content quality  
    📊 Calculate readability scores  
    🔍 Detect duplicate content  
    📈 Extract key insights  
    💡 Provide SEO recommendations
    """)
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    
    similarity_threshold = st.slider(
        "Duplicate Detection Threshold",
        min_value=0.5,
        max_value=0.95,
        value=0.80,
        step=0.05,
        help="Higher values = stricter duplicate detection"
    )
    
    show_advanced = st.checkbox("Show Advanced Metrics", value=False)
    
    st.markdown("---")
    st.markdown("### 📊 Dataset Statistics")
    
    if df_historical is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📄 Total Pages", len(df_historical))
        with col2:
            st.metric("📝 Avg Words", f"{df_historical['word_count'].mean():.0f}")
        
        # Quality distribution
        if 'quality_label' in df_historical.columns:
            st.markdown("**Quality Distribution:**")
            quality_counts = df_historical['quality_label'].value_counts()
            for label, count in quality_counts.items():
                st.write(f"{label}: {count}")
    
    st.markdown("---")
    st.markdown("### 📜 Analysis History")
    if st.session_state.analysis_history:
        st.write(f"Total analyzed: {len(st.session_state.analysis_history)}")
        if st.button("Clear History"):
            st.session_state.analysis_history = []
            st.rerun()
    else:
        st.write("No analysis yet")

# Main content - Enhanced tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Single URL Analysis", 
    "📊 Batch Analysis", 
    "📈 Analytics Dashboard",
    "💡 SEO Tips"
])

# Tab 1: Enhanced Single URL Analysis
with tab1:
    st.markdown("### 🔍 Analyze Single URL")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        url_input = st.text_input(
            "Enter URL to analyze:",
            placeholder="https://example.com/amazing-article",
            label_visibility="collapsed"
        )
    
    with col2:
        analyze_button = st.button("🚀 Analyze Now", type="primary", use_container_width=True)
    
    # Quick examples
    with st.expander("📌 Try These Example URLs"):
        example_urls = [
            "https://www.nytimes.com/",
            "https://techcrunch.com/",
            "https://medium.com/"
        ]
        for url in example_urls:
            if st.button(url, key=f"example_{url}"):
                url_input = url
                analyze_button = True
    
    if analyze_button and url_input:
        with st.spinner("🔄 Analyzing content... Please wait..."):
            try:
                # Scrape and parse
                html_content = scrape_url(url_input)
                
                if not html_content:
                    st.error("❌ Failed to fetch content from URL. Please check the URL and try again.")
                else:
                    parsed = parse_html_content(html_content)
                    
                    if parsed['word_count'] == 0:
                        st.error("❌ No content extracted from page. The page might be empty or blocked.")
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
                        
                        # Get insights
                        insights = get_quality_insights(features, prediction)
                        
                        # Add to history
                        st.session_state.analysis_history.append({
                            'url': url_input,
                            'quality': prediction['quality_label'],
                            'word_count': features['word_count'],
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        
                        # Success message
                        st.success("✅ Analysis Complete!")
                        
                        # Quality Badge
                        quality_class = f"quality-{prediction['quality_label'].lower()}"
                        st.markdown(
                            f'<div class="quality-badge {quality_class}">Quality: {prediction["quality_label"]}</div>',
                            unsafe_allow_html=True
                        )
                        
                        # Main Metrics Row
                        st.markdown("### 📊 Key Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="feature-icon">📝</div>
                                <div class="metric-value">{features['word_count']}</div>
                                <div class="metric-label">Words</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="feature-icon">📄</div>
                                <div class="metric-value">{features['sentence_count']}</div>
                                <div class="metric-label">Sentences</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="feature-icon">📖</div>
                                <div class="metric-value">{features['flesch_reading_ease']:.1f}</div>
                                <div class="metric-label">Readability</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col4:
                            confidence = max(prediction['confidence'].values()) * 100
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="feature-icon">🎯</div>
                                <div class="metric-value">{confidence:.0f}%</div>
                                <div class="metric-label">Confidence</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Page Information
                        with st.container():
                            st.markdown("### 📄 Page Details")
                            st.markdown(f"**Title:** {parsed['title']}")
                            st.markdown(f"**URL:** [{url_input}]({url_input})")
                            st.markdown(f"**Analysis Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        # Quality Assessment
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 🎯 Confidence Scores")
                            # Create gauge chart for confidence
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number+delta",
                                value = max(prediction['confidence'].values()) * 100,
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "Quality Confidence"},
                                gauge = {
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': "#667eea"},
                                    'steps': [
                                        {'range': [0, 50], 'color': "#fee140"},
                                        {'range': [50, 75], 'color': "#f5576c"},
                                        {'range': [75, 100], 'color': "#38ef7d"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 90
                                    }
                                }
                            ))
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show all confidence scores
                            for label, score in sorted(prediction['confidence'].items(), key=lambda x: x[1], reverse=True):
                                st.progress(score, text=f"{label}: {score:.1%}")
                        
                        with col2:
                            st.markdown("### 💡 Content Insights")
                            for insight in insights:
                                if "✅" in insight:
                                    st.success(insight)
                                elif "⚠️" in insight:
                                    st.warning(insight)
                                else:
                                    st.info(insight)
                        
                        # Keywords Section
                        if features.get('top_keywords'):
                            st.markdown("### 🔑 Top Keywords")
                            keywords = features['top_keywords'].split('|')
                            
                            # Display as colorful tags
                            keyword_html = " ".join([
                                f'<span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 0.3rem 0.8rem; border-radius: 1rem; margin: 0.2rem; display: inline-block;">{kw}</span>'
                                for kw in keywords
                            ])
                            st.markdown(keyword_html, unsafe_allow_html=True)
                        
                        # Advanced Metrics
                        if show_advanced:
                            st.markdown("### 📈 Advanced Metrics")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                avg_sentence_length = features['word_count'] / features['sentence_count'] if features['sentence_count'] > 0 else 0
                                st.metric("Avg Sentence Length", f"{avg_sentence_length:.1f} words")
                            
                            with col2:
                                st.metric("Thin Content", "Yes" if features['word_count'] < 500 else "No")
                            
                            with col3:
                                word_density = features['word_count'] / features['sentence_count'] if features['sentence_count'] > 0 else 0
                                st.metric("Word Density", f"{word_density:.1f}")
                        
                        # Duplicate Detection
                        if similar:
                            st.markdown("### 🔄 Duplicate Content Detected")
                            st.warning(f"⚠️ Found {len(similar)} similar page(s) - This may impact SEO!")
                            
                            # Create similarity chart
                            if len(similar) > 0:
                                similarity_df = pd.DataFrame(similar[:5])
                                fig = px.bar(
                                    similarity_df,
                                    x='similarity',
                                    y='url',
                                    orientation='h',
                                    title='Similarity Scores',
                                    color='similarity',
                                    color_continuous_scale='Reds'
                                )
                                fig.update_layout(height=300)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            for item in similar[:5]:
                                with st.expander(f"🔗 Similarity: {item['similarity']:.1%}"):
                                    st.write(f"**URL:** [{item['url']}]({item['url']})")
                                    st.write(f"**Similarity Score:** {item['similarity']:.4f}")
                        else:
                            st.markdown("### ✅ Unique Content")
                            st.success("🎉 No duplicate content detected! This is great for SEO.")
                        
                        # Content Preview
                        with st.expander("📝 Content Preview (First 500 characters)"):
                            st.text_area(
                                "Content",
                                value=parsed['body_text'][:500] + "...",
                                height=200,
                                disabled=True,
                                label_visibility="collapsed"
                            )
                        
                        # SEO Recommendations
                        st.markdown("### 🎯 SEO Recommendations")
                        recommendations = []
                        
                        if features['word_count'] < 300:
                            recommendations.append("📝 **Content Length:** Add at least 200 more words for better SEO ranking")
                        elif features['word_count'] < 500:
                            recommendations.append("📝 **Content Length:** Consider expanding to 500+ words")
                        else:
                            recommendations.append("✅ **Content Length:** Good! Keep creating quality content")
                        
                        if features['flesch_reading_ease'] < 30:
                            recommendations.append("📖 **Readability:** Simplify language for better user engagement")
                        elif features['flesch_reading_ease'] > 70:
                            recommendations.append("📖 **Readability:** Ensure professional tone is maintained")
                        else:
                            recommendations.append("✅ **Readability:** Perfect balance!")
                        
                        if similar:
                            recommendations.append("⚠️ **Duplicate Content:** Rewrite or consolidate similar pages")
                        else:
                            recommendations.append("✅ **Uniqueness:** Original content detected!")
                        
                        for i, rec in enumerate(recommendations, 1):
                            st.markdown(f"{i}. {rec}")
                        
                        # Download Report Button
                        report_data = {
                            'URL': url_input,
                            'Title': parsed['title'],
                            'Quality': prediction['quality_label'],
                            'Word Count': features['word_count'],
                            'Sentences': features['sentence_count'],
                            'Readability': features['flesch_reading_ease'],
                            'Keywords': features.get('top_keywords', ''),
                            'Duplicates Found': len(similar),
                            'Analysis Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        
                        report_df = pd.DataFrame([report_data])
                        csv = report_df.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Download Full Report (CSV)",
                            data=csv,
                            file_name=f"seo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.write("Please try again or contact support if the issue persists.")

# Tab 2: Enhanced Batch Analysis
with tab2:
    st.markdown("### 📊 Batch URL Analysis")
    
    st.info("💡 **Tip:** Upload a CSV file with a 'url' column to analyze multiple URLs at once!")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="CSV must contain a column named 'url'"
    )
    
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            
            if 'url' not in df_batch.columns:
                st.error("❌ CSV must contain a 'url' column")
            else:
                st.success(f"✅ Found {len(df_batch)} URLs to analyze")
                
                with st.expander("📋 Preview URLs"):
                    st.dataframe(df_batch.head(10), use_container_width=True)
                
                # Batch settings
                col1, col2 = st.columns(2)
                with col1:
                    max_urls = st.number_input(
                        "Maximum URLs to analyze",
                        min_value=1,
                        max_value=len(df_batch),
                        value=min(10, len(df_batch))
                    )
                with col2:
                    delay = st.slider("Delay between requests (seconds)", 0.5, 5.0, 1.5)
                
                if st.button("🚀 Start Batch Analysis", type="primary", use_container_width=True):
                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    df_subset = df_batch.head(int(max_urls))
                    
                    for idx, row in df_subset.iterrows():
                        url = row['url']
                        status_text.text(f"🔄 Processing {idx+1}/{len(df_subset)}: {url[:50]}...")
                        
                        try:
                            import time
                            html_content = scrape_url(url)
                            if html_content:
                                parsed = parse_html_content(html_content)
                                features = extract_features(parsed['body_text'])
                                prediction = predict_quality(model_data, features)
                                
                                results.append({
                                    'URL': url,
                                    'Title': parsed['title'],
                                    'Word Count': features['word_count'],
                                    'Readability': round(features['flesch_reading_ease'], 2),
                                    'Quality': prediction['quality_label'],
                                    'Confidence': f"{max(prediction['confidence'].values()):.1%}",
                                    'Thin Content': 'Yes' if features['word_count'] < 500 else 'No',
                                    'Status': '✅ Success'
                                })
                            else:
                                results.append({
                                    'URL': url,
                                    'Title': 'N/A',
                                    'Word Count': 0,
                                    'Readability': 0,
                                    'Quality': 'Error',
                                    'Confidence': '0%',
                                    'Thin Content': 'Yes',
                                    'Status': '❌ Failed'
                                })
                            time.sleep(delay)
                        except Exception as e:
                            results.append({
                                'URL': url,
                                'Title': 'N/A',
                                'Word Count': 0,
                                'Readability': 0,
                                'Quality': 'Error',
                                'Confidence': '0%',
                                'Thin Content': 'Yes',
                                'Status': f'❌ Error: {str(e)[:50]}'
                            })
                        
                        progress_bar.progress((idx + 1) / len(df_subset))
                    
                    status_text.text("✅ Batch analysis complete!")
                    
                    # Display results
                    df_results = pd.DataFrame(results)
                    st.markdown("### 📊 Analysis Results")
                    st.dataframe(df_results, use_container_width=True)
                    
                    # Summary Statistics
                    st.markdown("### 📈 Summary Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Analyzed", len(df_results))
                    
                    with col2:
                        high_quality = (df_results['Quality'] == 'High').sum()
                        st.metric("High Quality", high_quality, 
                                 delta=f"{high_quality/len(df_results)*100:.1f}%")
                    
                    with col3:
                        thin_content = (df_results['Thin Content'] == 'Yes').sum()
                        st.metric("Thin Content", thin_content,
                                 delta=f"{thin_content/len(df_results)*100:.1f}%",
                                 delta_color="inverse")
                    
                    with col4:
                        avg_words = df_results['Word Count'].mean()
                        st.metric("Avg Words", f"{avg_words:.0f}")
                    
                    # Charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Quality distribution
                        quality_counts = df_results['Quality'].value_counts()
                        fig = px.pie(
                            values=quality_counts.values,
                            names=quality_counts.index,
                            title='Quality Distribution',
                            color_discrete_sequence=px.colors.sequential.RdBu
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Word count distribution
                        fig = px.histogram(
                            df_results,
                            x='Word Count',
                            title='Word Count Distribution',
                            nbins=20,
                            color_discrete_sequence=['#667eea']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Download results
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Complete Results (CSV)",
                        data=csv,
                        file_name=f"batch_seo_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# Tab 3: Analytics Dashboard
with tab3:
    st.markdown("### 📈 Analytics Dashboard")
    
    if st.session_state.analysis_history:
        history_df = pd.DataFrame(st.session_state.analysis_history)
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Analyses", len(history_df))
        
        with col2:
            high_quality_pct = (history_df['quality'] == 'High').sum() / len(history_df) * 100
            st.metric("High Quality %", f"{high_quality_pct:.1f}%")
        
        with col3:
            st.metric("Avg Word Count", f"{history_df['word_count'].mean():.0f}")
        
        with col4:
            st.metric("Latest Analysis", history_df.iloc[-1]['timestamp'])
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Quality trend
            fig = px.line(
                history_df,
                x='timestamp',
                y='word_count',
                title='Word Count Trend Over Time',
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Quality distribution
            quality_counts = history_df['quality'].value_counts()
            fig = px.bar(
                x=quality_counts.index,
                y=quality_counts.values,
                title='Quality Distribution',
                color=quality_counts.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent analyses
        st.markdown("### 📋 Recent Analyses")
        st.dataframe(history_df.tail(10), use_container_width=True)
        
    else:
        st.info("📊 No analysis history yet. Start analyzing URLs to see your dashboard!")

# Tab 4: SEO Tips - CONTINUED AND COMPLETED
with tab4:
    st.markdown("### 💡 SEO Best Practices & Tips")
    
    tips_categories = {
        "📝 Content Quality": [
            "Aim for at least 300 words, ideally 500-1000 words for better ranking",
            "Write unique, original content - avoid copying from other sources",
            "Use clear, concise language that your target audience understands",
            "Break content into short paragraphs (2-3 sentences max)",
            "Include relevant keywords naturally - don't stuff them!"
        ],
        "📖 Readability": [
            "Target a Flesch Reading Ease score between 50-70",
            "Use short sentences (15-20 words on average)",
            "Avoid jargon unless writing for technical audiences",
            "Use active voice instead of passive voice",
            "Include subheadings to break up long text"
        ],
        "🔍 SEO Technical": [
            "Include target keywords in title, headings, and first paragraph",
            "Use descriptive, keyword-rich URLs",
            "Add internal links to related content on your site",
            "Optimize images with alt text and compression",
            "Ensure mobile-friendly design and fast loading times"
        ],
        "🎯 Content Structure": [
            "Start with a compelling introduction",
            "Use H1 for main title, H2 for sections, H3 for subsections",
            "Include bullet points and numbered lists for scanability",
            "Add relevant images, videos, or infographics",
            "End with a clear call-to-action or conclusion"
        ],
        "🔄 Duplicate Content": [
            "Always create original content - never copy-paste",
            "Use canonical tags for similar pages",
            "Consolidate thin or duplicate pages",
            "Rewrite product descriptions instead of using manufacturer text",
            "Use 301 redirects for removed duplicate pages"
        ],
        "📊 Performance Metrics": [
            "Monitor your content's performance in Google Search Console",
            "Track organic traffic, bounce rate, and time on page",
            "A/B test different headlines and content formats",
            "Update old content regularly to keep it fresh",
            "Analyze competitor content to identify gaps"
        ]
    }
    
    # Display tips in expandable cards
    for category, tips in tips_categories.items():
        with st.expander(f"**{category}**", expanded=False):
            st.markdown('<div class="tip-card">', unsafe_allow_html=True)
            for i, tip in enumerate(tips, 1):
                st.markdown(f"{i}. {tip}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional Resources Section
    st.markdown("---")
    st.markdown("### 📚 Additional Resources")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>🎓 Learning Resources</h4>
            <ul>
                <li><strong>Google Search Central:</strong> Official SEO guidelines</li>
                <li><strong>Moz Beginner's Guide:</strong> Comprehensive SEO tutorial</li>
                <li><strong>Ahrefs Blog:</strong> Advanced SEO strategies</li>
                <li><strong>Search Engine Journal:</strong> Latest SEO news</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>🛠️ Recommended Tools</h4>
            <ul>
                <li><strong>Google Search Console:</strong> Monitor search performance</li>
                <li><strong>Google Analytics:</strong> Track user behavior</li>
                <li><strong>Screaming Frog:</strong> Technical SEO audits</li>
                <li><strong>Hemingway Editor:</strong> Improve readability</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick SEO Checklist
    st.markdown("---")
    st.markdown("### ✅ Quick SEO Checklist")
    
    checklist_items = [
        "Content is at least 300 words (ideally 500+)",
        "Title tag includes target keyword and is under 60 characters",
        "Meta description is compelling and under 160 characters",
        "URL is short, descriptive, and includes keyword",
        "H1 tag is used once and includes main keyword",
        "Content includes relevant keywords naturally (1-2% density)",
        "Images have descriptive alt text",
        "Internal links to related content (3-5 links)",
        "External links to authoritative sources (1-3 links)",
        "Content is mobile-friendly and loads quickly",
        "No duplicate content issues",
        "Flesch Reading Ease score between 50-70",
        "Content provides unique value to readers"
    ]
    
    # Create two columns for checklist
    col1, col2 = st.columns(2)
    mid_point = len(checklist_items) // 2
    
    with col1:
        for item in checklist_items[:mid_point]:
            st.checkbox(item, key=f"check_{item[:20]}")
    
    with col2:
        for item in checklist_items[mid_point:]:
            st.checkbox(item, key=f"check_{item[:20]}")
    
    # Content Quality Scoring Guide
    st.markdown("---")
    st.markdown("### 🎯 Understanding Quality Scores")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="tip-card">
            <div class="tip-title" style="color: #38ef7d;">✅ High Quality</div>
            <ul>
                <li>500+ words of original content</li>
                <li>Good readability (50-70 score)</li>
                <li>Proper structure with headings</li>
                <li>Relevant keywords naturally used</li>
                <li>No duplicate content</li>
                <li>Engaging and informative</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="tip-card">
            <div class="tip-title" style="color: #f5576c;">⚠️ Medium Quality</div>
            <ul>
                <li>300-500 words</li>
                <li>Decent readability</li>
                <li>Basic structure present</li>
                <li>Some optimization needed</li>
                <li>Room for improvement</li>
                <li>Could be more detailed</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="tip-card">
            <div class="tip-title" style="color: #fee140;">❌ Low Quality</div>
            <ul>
                <li>Under 300 words (thin content)</li>
                <li>Poor readability</li>
                <li>Lack of structure</li>
                <li>Duplicate or scraped content</li>
                <li>Keyword stuffing</li>
                <li>Needs major revision</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Common SEO Mistakes
    st.markdown("---")
    st.markdown("### ⚠️ Common SEO Mistakes to Avoid")
    
    mistakes = [
        {
            "mistake": "**Keyword Stuffing**",
            "description": "Overusing keywords unnaturally. Aim for 1-2% keyword density.",
            "icon": "🚫"
        },
        {
            "mistake": "**Duplicate Content**",
            "description": "Publishing similar content across multiple pages hurts rankings.",
            "icon": "🔄"
        },
        {
            "mistake": "**Thin Content**",
            "description": "Pages with less than 300 words provide little value to users.",
            "icon": "📄"
        },
        {
            "mistake": "**Ignoring Mobile Users**",
            "description": "Over 60% of searches are mobile. Ensure responsive design.",
            "icon": "📱"
        },
        {
            "mistake": "**Slow Page Speed**",
            "description": "Pages that load slowly have higher bounce rates.",
            "icon": "🐌"
        },
        {
            "mistake": "**No Internal Linking**",
            "description": "Internal links help users navigate and distribute page authority.",
            "icon": "🔗"
        }
    ]
    
    for mistake in mistakes:
        st.markdown(f"""
        <div class="info-card">
            <h4>{mistake['icon']} {mistake['mistake']}</h4>
            <p>{mistake['description']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Pro Tips Section
    st.markdown("---")
    st.markdown("### 🌟 Pro Tips for Better SEO")
    
    pro_tips = [
        "**Update Old Content:** Refresh your top-performing pages every 6-12 months",
        "**Answer Questions:** Use FAQ sections to target featured snippets",
        "**Use Schema Markup:** Help search engines understand your content better",
        "**Optimize for Voice Search:** Use natural language and question formats",
        "**Build Topic Clusters:** Create pillar pages with related supporting content",
        "**Monitor Core Web Vitals:** Google uses these for ranking (LCP, FID, CLS)",
        "**Earn Backlinks:** Quality backlinks from authoritative sites boost rankings",
        "**Use Long-Tail Keywords:** Less competitive and more specific to user intent"
    ]
    
    for tip in pro_tips:
        st.info(f"💡 {tip}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <h3 style="color: #667eea;">🚀 SEO Content Analyzer Pro</h3>
    <p>Built with ❤️ using Streamlit | Powered by Machine Learning</p>
    <p style="font-size: 0.9rem; margin-top: 1rem;">
        <strong>Features:</strong> Quality Analysis • Duplicate Detection • Readability Scoring • Batch Processing
    </p>
    <p style="font-size: 0.8rem; color: #999; margin-top: 1rem;">
        v2.0.0 | © 2024 | For support, contact your administrator
    </p>
</div>
""", unsafe_allow_html=True)