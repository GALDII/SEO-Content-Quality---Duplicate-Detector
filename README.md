Here is a comprehensive README file for your project, based on the code, notebook, and project structure you provided.

---

# 🚀 SEO Content Quality & Duplicate Detector

This project is an AI-powered Streamlit dashboard designed to analyze web page content for SEO. It provides two core functionalities:

1.  **Content Quality Scoring:** Uses a machine learning model to predict content quality (High, Medium, or Low) based on features like word count, readability, and sentence structure.
2.  **Duplicate Content Detection:** Uses `sentence-transformers` to generate semantic embeddings and identify a-list pages with similar content from a historical database.

The application includes a rich, multi-tab interface for single URL analysis, batch CSV analysis, a history dashboard, and general SEO tips.

## ✨ Features

- **Single URL Analysis:** Get an instant, in-depth report for any URL.
- **Batch Analysis:** Upload a CSV file of URLs to analyze multiple pages at once.
- **ML-Powered Quality Score:** Predicts content quality (High, Medium, Low) using a trained Random Forest model.
- **Semantic Duplicate Detection:** Compares a new URL's content against a historical database of page embeddings to flag potential duplicates.
- **Key Metric Extraction:** Automatically parses and calculates:
  - Word Count
  - Sentence Count
  - Flesch Reading Ease
  - Average Word Length
  - Unique Word Ratio
- **Actionable Insights:** Provides plain-English recommendations based on the analysis (e.g., "Content is too short," "Readability is well-balanced").
- **Interactive Dashboard:** Features Plotly charts for quality distribution, confidence scores, and similarity analysis.
- **Complete Training Pipeline:** A Jupyter Notebook (`seo_pipeline.ipynb`) is included to show the entire process of data scraping, feature engineering, model training, and data-file generation.

## 📁 Project Structure

```
SEO-CONTENT-DETECTOR/
├── data/                    # Stores all data files
│   ├── data.csv             # (Input) Raw URLs/content for training
│   ├── extracted_content.csv # (Output) Parsed text from raw data
│   ├── features.csv         # (Output) Features & embeddings for all pages
│   └── duplicates.csv       # (Output) Report of duplicate pairs
├── models/                  # Stores the trained ML model
│   └── quality_model.pkl    # (Output) Trained RandomForest model
├── notebooks/
│   └── seo_pipeline.ipynb   # Jupyter notebook to run the full pipeline
├── streamlit_app/           # Main application folder
│   ├── models/              # (Optional) Can hold a copy of the model
│   │   └── quality_model.pkl
│   ├── utils/               # Helper modules for the app
│   │   ├── __init__.py
│   │   ├── features.py      # Functions to extract text features
│   │   ├── parser.py        # Functions for scraping and HTML parsing
│   │   └── scorer.py        # Functions to load model, predict, find similarity
│   ├── app.py               # The main Streamlit application file
├── .gitignore
├── packages.txt             # System-level dependencies for Streamlit Cloud
├── README.md                # This file
└── requirements.txt         # Python dependencies
```

## 🛠️ How It Works

The project is split into two main parts: the training pipeline and the Streamlit application.

### 1\. Training Pipeline (`notebooks/seo_pipeline.ipynb`)

This notebook is responsible for creating the model and data files the Streamlit app depends on.

1.  **Load Data:** Reads a list of URLs from `data/data.csv`.
2.  **Parse & Extract:** Scrapes each URL (using `requests` and `BeautifulSoup`) to extract the main body text. Saves the clean text to `data/extracted_content.csv`.
3.  **Feature Engineering:** For each page's text, it calculates:
    - **Text Stats:** Word count, sentence count, Flesch reading ease, average word length (using `textstat` and `nltk`).
    - **Embeddings:** Generates 384-dimension semantic vector embeddings using `sentence-transformers ('all-MiniLM-L6-v2')`.
    - All features and embeddings are saved to `data/features.csv`.
4.  **Duplicate Analysis:** A cosine similarity matrix is computed for all embeddings. Pairs with similarity above a set threshold (e.g., 0.80) are saved to `data/duplicates.csv`.
5.  **Model Training:**
    - Synthetic labels (`High`, `Medium`, `Low`) are created based on rules (e.g., word count \> 1500 = High).
    - A `RandomForestClassifier` is trained on the text stat features (not the embeddings) to predict these labels.
6.  **Save Model:** The final trained model is saved to `models/quality_model.pkl`.

### 2\. Streamlit App (`streamlit_app/app.py`)

This is the real-time inference engine that provides the user interface.

1.  **Load Assets:** On startup, the app loads `models/quality_model.pkl` and the `data/features.csv` file (to use its embeddings for duplicate checking).
2.  **Get Input:** The user provides a URL.
3.  **Analyze (Real-time):**
    - The app scrapes the URL using `utils/parser.py`.
    - It extracts all text features (word count, readability, etc.) using `utils/features.py`.
    - It generates a _new_ embedding for the scraped text.
4.  **Predict:**
    - **Quality:** The extracted features are fed into the loaded model to get a `High/Medium/Low` quality prediction (`utils/scorer.py`).
    - **Duplicates:** The _new_ embedding is compared against all historical embeddings loaded from `features.csv` to find the most similar pages (`utils/scorer.py`).
5.  **Display:** All results, insights, and charts are displayed on the interactive dashboard.

## 🚀 Setup and Installation

### Prerequisites

- Python 3.9+
- `pip` and `virtualenv`

### 1\. Clone & Set Up Environment

```bash
# Clone the repository
git clone https://github.com/your-username/SEO-CONTENT-DETECTOR.git
cd SEO-CONTENT-DETECTOR

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2\. Install Dependencies

Install all required Python packages.

```bash
pip install -r requirements.txt
```

The app also uses NLTK data. The scripts (`features.py` and the notebook) will attempt to download `punkt` and `stopwords` automatically.

## ▶️ How to Run

There are two steps to run this project:

### Step 1: Run the Training Pipeline (Mandatory)

Before you can use the Streamlit app, you **must** generate the model and historical data files.

1.  Launch Jupyter Notebook:
    ```bash
    jupyter-notebook
    ```
2.  Open `notebooks/seo_pipeline.ipynb`.
3.  Run all cells in the notebook from top to bottom.

This will create the following essential files:

- `models/quality_model.pkl`
- `data/features.csv`
- `data/extracted_content.csv`
- `data/duplicates.csv`

### Step 2: Run the Streamlit App

Once the files from Step 1 are generated, you can run the web application.

1.  Navigate to the `streamlit_app` directory:
    ```bash
    cd streamlit_app
    ```
2.  Run the app using Streamlit:
    ```bash
    streamlit run app.py
    ```
3.  Open the local URL (e.g., `http://localhost:8501`) in your web browser.

## 🔧 Technologies Used

- **Web Framework:** Streamlit
- **Machine Learning:** Scikit-learn
- **NLP & Embeddings:** Sentence-Transformers, PyTorch, NLTK, Textstat
- **Data Handling:** Pandas, NumPy
- **Web Scraping:** Requests, BeautifulSoup4 (lxml)
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Model Persistence:** Joblib (or Pickle)
